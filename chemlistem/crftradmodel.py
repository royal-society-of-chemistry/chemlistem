import time
import io
import sys
import os
import random
import json
import numpy as np
import csv
import shutil

from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import Input, concatenate, CuDNNLSTM
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.regularizers import l1_l2
from datetime import datetime

from keras_contrib.layers.crf import CRF

from .featurizer import Featurizer
from chemtok import ChemTokeniser
from .utils import tobits, sobie_scores_to_char_ents, get_file
from .corpusreader import CorpusReader

defaultmodel = None

def get_trad_model():
    """
    Gets the default pre-trained traditional model, loading if necessary.
    
    Returns:
        An TradModel
    """
    global defaultmodel
    if defaultmodel is not None: return defaultmodel
    tm = TradModel()
    jf = get_file("default_tradmodel_0.0.1.json")
    hf = get_file("default_tradmodel_0.0.1.h5")    
    tm.load(jf, hf)
    defaultmodel = tm
    return defaultmodel
    
class CRFTradModel(object):
    """
    A "traditional" model for chemical named entity recognition - works in a similar manner to CRF-based models,
    with tokenisation and a rich feature set.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        pass
    
    def _str_to_seq(self, str):
        seq = {"tokens": [], "bio": [], "tokstart": [], "tokend": [], "str": str}
        ct = ChemTokeniser(str, clm=True)
        for t in ct.tokens:
            seq["tokens"].append(t.value)
            seq["tokstart"].append(t.start)
            seq["tokend"].append(t.end)
        return seq
    
    def _prepare_seqs(self, seqs, verbose=True):
        """
        Add features to sequences.
        """
        if verbose: print("Number words at", datetime.now(), file=sys.stderr)
        # Tokens to numbers, for use with embeddings
        for seq in seqs:
            seq["wordn"] = [self.tokdict[i] if i in self.tokdict else self.tokdict["*UNK*"] for i in seq["tokens"]]
        if verbose: print("Generate features at", datetime.now(), file=sys.stderr)
        # Words to name-internal features
        for seq in seqs:
            seq["ni"] = np.array([self.fzr.num_feats_for_tok(i) for i in seq["tokens"]])
        if verbose: print("Generated features at", datetime.now(), file=sys.stderr)
        
        # Take a note of how many name-internal features there are
        self.nilen = len(seqs[0]["ni"][0])
    
    def train(self, textfile, annotfile, glovefile, runname):
        """
        Train a new TradModel.
        
        May optionally use pre-trained embedding vectors from GloVe: https://nlp.stanford.edu/projects/glove/
        
        Use the 6B model, and the 300d vectors - the filename should be "glove.6B.300d.txt". This is not
        distributed with chemlistem - you will need to get it yourself.
        
        This produces two important files:
        
        tradmodel_$RUNNAME.h5 - the keras model itself
        tradmodel_$RUNNAME.json - various bits of auxilliary information
        
        These consititute the trained model.
        
        It also produces several files named:
        
        epoch_$EPOCHNUM_$RUNAME.h5
        
        These are the keras models for each epoch (the auxilliary information doesn't change).
        
        Args:
            textfile: the filename of the file containing the BioCreative training text - e.g. "BioCreative V.5 training set.txt"
            annotfile: the filename of the containing the BioCreative training annotations - e.g. "CEMP_BioCreative V.5 training set annot.tsv"
            glovefile: None, or the filename of the glove file - e.g. "glove.6B.300d.txt"
            runname: a string, part of the output filenames.
        """
        # Get training and test sequences
        cr = CorpusReader(textfile, annotfile)
        train = cr.trainseqs
        test = cr.testseqs

        seqs = train+test
        
        # Initialise some stuff
        toklist = []
        tokdict = {}
        tokcounts = {}
        labels = set()
        self.toklist = toklist
        self.tokdict = tokdict
        self.tokcounts = tokcounts
        self.fzr = None # Make later
        self.lablist = None # Do later
        self.labdict = None # Do later
        self.model = None # Do later
        
        # Count tokens in training data....
        for seq in train:
            for tok in seq["tokens"]:
                if tok not in tokcounts: tokcounts[tok] = 0
                tokcounts[tok] += 1
        
        # and keep those that occur more than twice
        for tok in list(tokcounts.keys()):
            if tokcounts[tok] > 2:
                tokdict[tok] = len(toklist)
                toklist.append(tok)
        tokdict["*UNK*"] = len(toklist)
        toklist.append("*UNK*")
        
        ohn = len(toklist)

        # Initialise embeddings using GloVe if present
        em = None
        if glovefile is not None:
            t = time.time()
            ei = {}
            f = open(glovefile, "r", encoding="utf-8")
            for l in f:
                ll = l.strip().split()
                w = ll[0]
                c = np.asarray(ll[1:], dtype='float32')
                ei[w] = c
            em = np.zeros((ohn, 300))
            for i in range(ohn):
                if toklist[i] in ei: em[i] = ei[toklist[i]]
            print("Embeddings read in:", time.time() - t, file=sys.stderr)

        # Collect labels for tokens
        for seq in seqs:
            for i in seq["bio"]: labels.add(i)
        lablist = sorted(labels)
        lablist.reverse()
        labdict = {lablist[i]:i for i in range(len(lablist))}
        self.lablist = lablist
        self.labdict = labdict
        
        # Convert SOBIE tags to numbers
        for seq in seqs:
            seq["bion"] = [labdict[i] for i in seq["bio"]]

        # Build the "featurizer" which generates token-internal features
        print("Make featurizer at", datetime.now(), file=sys.stderr)
        fzr = Featurizer(train)
        self.fzr = fzr

        # Marshal features for each token
        self._prepare_seqs(seqs)
        
        # Gather together sequences by length
        print("Make train dict at", datetime.now(), file=sys.stderr)
        
        train_l_d = {}
        for seq in train:
            l = len(seq["tokens"])
            if l not in train_l_d: train_l_d[l] = []
            train_l_d[l].append(seq)
                    
        # Set up the keras model
        il = Input(shape=(None, self.nilen))
        cl = Conv1D(256, 3, padding='same', activation='relu', input_shape=(None, self.nilen), name="conv")(il)
        cdl = Dropout(0.5)(cl)
        ei = Input(shape=(None,), dtype='int32')
        if em is not None:
            el = Embedding(ohn, 300, weights=[em])(ei)
        else:
            el = Embedding(ohn, 300)(ei)        
        ml = concatenate([cdl, el])
        bll = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001)), merge_mode="concat", name="lstm")(ml)
        blld = Dropout(0.5)(bll)
        #dl = TimeDistributed(Dense(len(lablist), activation="softmax"), name="output")(blld)
        #crf = CRF(len(lablist), learn_mode="join", test_mode="viterbi", sparse_target=False)
        crf = CRF(len(lablist), learn_mode="marginal", test_mode="marginal", sparse_target=False)
        crfl = crf(blld)
        model = Model(inputs=[ei, il], outputs=crfl )
        model.compile(loss=crf.loss_function, optimizer='rmsprop', metrics=[crf.accuracy])
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model
        
        # Serialize the auxilliary intformation
        outjo = {
            "tokdict": self.tokdict,
            "fzr": self.fzr.to_json_obj(),
            "lablist": self.lablist
            }
            
        print("Serialize at", datetime.now(), file=sys.stderr)
        jf = open("tradmodel_%s.json" % runname, "w", encoding="utf-8")
        json.dump(outjo, jf)
        jf.close()
        
        sizes = list(train_l_d)
        
        best_epoch = -1
        best_f = 0.0
        
        # OK, start actually training
        for epoch in range(20):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            # Train in batches of different sizes - randomize the order of sizes
            random.shuffle(sizes)
            tnt = 0
            for size in sizes: tnt += size * len(train_l_d[size])
            for size in sizes:
                if size == 1: continue # For unknown reasons we can't train on a single token
                batch = train_l_d[size]
                tx = np.array([seq["wordn"] for seq in batch])
                tx2 = np.array([seq["ni"] for seq in batch])
                ty = np.array([[tobits(i, len(lablist)) for i in seq["bion"]] for seq in batch])
                # This trains in mini-batches
                model.fit([tx, tx2], ty, verbose=0, epochs=1)
            print("Trained at", datetime.now(), file=sys.stderr)
            self.model.save("epoch_%s_%s.h5" % (epoch, runname))
            # Evaluate
            tp_all = 0
            fp_all = 0
            fn_all = 0
            for i in range(len(test)):
                
                enttype = None
                entstart = 0
                ts = test[i]
                ents = [("E", i[2], i[3]) for i in ts["ents"]]
                mm = model.predict([np.array([ts["wordn"]]), np.array([ts["ni"]])])[0]
                
                pseq = {}
                pseq["tokens"] = ts["tokens"]
                pseq["tokstart"] = ts["tokstart"]
                pseq["tokend"] = ts["tokend"]
                pseq["tagfeat"] = mm
                
                pents, pxe = sobie_scores_to_char_ents(pseq, 0.5, ts["ss"])
                tp = 0
                fp = 0
                fn = 0
                tofind = set(ents)
                for ent in pents:
                    if ent in tofind:
                        tp += 1
                    else:
                        fp += 1
                fn = len(tofind) - tp
                tp_all += tp
                fp_all += fp
                fn_all += fn
            
            f = (2*tp_all/(tp_all+tp_all+fp_all+fn_all))
            print("TP", tp_all, "FP", fp_all, "FN", fn_all, "F", f, "Precision", tp_all/(tp_all+fp_all), "Recall", tp_all/(tp_all+fn_all), file=sys.stderr)
            if f > best_f:
                print("Best so far", file=sys.stderr)
                best_f = f
                best_epoch = epoch
        
        # Pick the best model, and save it with a useful name
        if best_epoch > -1:
            shutil.copyfile("epoch_%s_%s.h5" % (best_epoch, runname), "tradmodel_%s.h5" % runname)

    def load(self, jfile, mfile):
        """
        Load in model data.
        
        Args:
            jfile: the filename of the .json file
            mfile: the filename of the .h5 file
        """
        jf = open(jfile, "r", encoding="utf-8")
        jo = json.load(jf)
        jf.close()
        self.tokdict = jo["tokdict"]
        self.lablist = jo["lablist"]
        self.fzr = Featurizer(None, jo["fzr"])
        print("Auxillary information read at", datetime.now(), file=sys.stderr)
        self.model = load_model(mfile)
        print("Traditional Model read at", datetime.now(), file=sys.stderr)
    
    def process(self, str, threshold=0.5, domonly=True):
        """
        Find named entities in a string.
        
        Entities are returned as tuples:
        (start_charater_position, end_character_position, string, score, is_dominant)
        
        Entities are dominant if they are not partially or wholly overlapping with a higher-scoring entity.
        
        Args:
            str: the string to find entities in.
            threshold: the minimum score for entities.
            domonly: if True, discard non-dominant entities.
        """
        results = []
        if len(str) == 0: return results
        seq = self._str_to_seq(str)
        if len(seq["tokens"]) == 0: return results
        self._prepare_seqs([seq], False) # Not verbose
        mm = self.model.predict([np.array([seq["wordn"]]), np.array([seq["ni"]])])[0]
        seq["tagfeat"] = mm
        pents, pxe = sobie_scores_to_char_ents(seq, threshold, str)
        if domonly:
            pents = [i for i in pents if pxe[i]["dom"]]
        for ent in pents:
            results.append((ent[1], ent[2], str[ent[1]:ent[2]], pxe[ent]["score"], pxe[ent]["dom"]))
        return results

    
