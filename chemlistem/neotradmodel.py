import time
import io
import sys
import os
import random
import json
import numpy as np
import csv
import shutil
import re
import math
from collections import defaultdict

from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers import Input, concatenate, CuDNNLSTM
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l1_l2
from datetime import datetime
import keras.backend as K

from .featurizer import Featurizer
from chemtok import ChemTokeniser
from .utils import tobits, sobie_scores_to_char_ents, get_file
from .corpusreader import CorpusReader
from .evaltools import internal_eval

#defaultmodel = None
#
#def get_trad_model():
#    """
#    Gets the default pre-trained traditional model, loading if necessary.
#    
#    Returns:
#        An TradModel
#    """
#    global defaultmodel
#    if defaultmodel is not None: return defaultmodel
#    tm = TradModel()
#    jf = get_file("default_tradmodel_0.0.1.json")
#    hf = get_file("default_tradmodel_0.0.1.h5")    
#    tm.load(jf, hf)
#    defaultmodel = tm
#    return defaultmodel
    
unkres = [
    ("num", re.compile("^-?[0-9]+(\\.[0-9]+)?$")),
    ("pc", re.compile("^-?[0-9]+(\\.[0-9]+)?%$"))
]
unkre_hascd = re.compile("^.*[0-9A-Za-z].*$")
unkre_hasc = re.compile("^.*[A-Za-z].*$")
unkre_haslc = re.compile("^.*[a-z].*$")
unkre_has2lc = re.compile("^.*[a-z][a-z].*$")
unkre_dandc = re.compile("^.*([A-Za-z].*[0-9]|[0-9].*[A-Za-z]).*$")
    
class NeoTradModel(object):
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
            seq["wordn"] = [self.tokdict[i] if i in self.tokdict else self.tokdict[self.classify_unk(i)] for i in seq["tokens"]]
            seq["worde"] = [self.ei[i] if i in self.ei else self.ei["unk"] for i in seq["tokens"]]
            #seq["wordn"] = [self.tokdict[i] if i in self.tokdict else self.tokdict["*UNK*"] for i in seq["tokens"]]
        if verbose: print("Generate features at", datetime.now(), file=sys.stderr)
        # Words to name-internal features
        for seq in seqs:
            seq["ni"] = np.array([self.fzr.num_feats_for_tok(i) for i in seq["tokens"]])
        if verbose: print("Generated features at", datetime.now(), file=sys.stderr)
        
        # Take a note of how many name-internal features there are
        self.nilen = len(seqs[0]["ni"][0])
    
    def classify_unk(self, s):
        if not self.do_classify_unk:
            return "*UNK*"
        if s in self.unkcache: return self.unkcache[s]
        unkt = "*UNK*"
        uu = None
        for ur in unkres:
            if ur[1].match(s):
                uu = ur[0]
                break
        if uu is not None:
            unkt = "*UNKre_%s*" % uu
        elif s.endswith("yl"):
            unkt = "*UNKyl*"
        elif s.endswith("ate"):
            unkt = "*UNKate*"
        elif s.endswith("ated"):
            unkt = "*UNKated*"
        elif s.endswith("ed"):
            unkt = "*UNKed*"
        elif s.endswith("ing"):
            unkt = "*UNKing*"
        elif s.endswith("ic"):
            unkt = "*UNKic*"
        elif s.endswith("tion"):
            unkt = "*UNKtion*"
        elif s.endswith("al"):
            unkt = "*UNKal*"
        elif s.endswith("ive"):
            unkt = "*UNKive*"
        elif s.endswith("ly"):
            unkt = "*UNKly*"
        elif s.endswith("ane"):
            unkt = "*UNKane*"
        elif s.endswith("ene"):
            unkt = "*UNKene*"
        elif s.endswith("yne"):
            unkt = "*UNKyne*"
        elif s.endswith("ine"):
            unkt = "*UNKine*"
        elif s.endswith("ase"):
            unkt = "*UNKase*"
        elif s.endswith("ble"):
            unkt = "*UNKble*"
        elif s.endswith("ity"):
            unkt = "*UNKity*"
        elif s.endswith("rgic"):
            unkt = "*UNKrgic*"
        elif s.endswith("ein"):
            unkt = "*UNKein*"
        elif s.endswith("in"):
            unkt = "*UNKin*"
        elif s.endswith("ose"):
            unkt = "*UNKose*"
        elif s.endswith("one"):
            unkt = "*UNKone*"
        elif s.endswith("dehyde"):
            unkt = "*UNKdehyde*"
        elif s.endswith("ol"):
            unkt = "*UNKol*"
        elif s.endswith("ide"):
            unkt = "*UNKide*"
        elif s.endswith("ite"):
            unkt = "*UNKite*"
        elif s.endswith("ium"):
            unkt = "*UNKium*"
        elif s.endswith("s"):
            unkt = "*UNKs*"
        #elif s.endswith("-"):
        #    unkt = "*UNK-*"
        #elif not unkre_hascd.match(s):
        #    unkt = "*UNK_nocd*"
        #elif not unkre_hasc.match(s):
        #    unkt = "*UNK_noc*"
        #elif not unkre_haslc.match(s):
        #    unkt = "*UNK_nolc*"
        #elif not unkre_has2lc.match(s):
        #    unkt = "*UNK_no2lc*"
        #if unkre_dandc.match(s):
        #    unkt = "*UNK_dandc*"
        self.unkcache[s] = unkt
        return unkt
            
    def get_ei(self, glovefile):
        ei = {}
        self.ei = ei
        ei["$PAD"] = np.array([0]*300)
        f = open(glovefile, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split()
            w = ll[0]
            c = np.asarray(ll[1:], dtype='float32')
            ei[w] = c
        if "unk" not in ei and "<unk>" in ei: ei["unk"] = ei["<unk>"]
        if "unk" not in ei: ei["unk"] = np.array([0]*300)
        if "*UNK*" not in ei: ei["*UNK*"] = ei["unk"]
    
    def train(self, textfile, annotfile, glovefile, runname, bonusfn=None, mbs=60000, do_classify_unk=True, ntype=0, extrachem=None):
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
        
        self.do_classify_unk = do_classify_unk
        
        if bonusfn is not None:
            bonusf = open(bonusfn, "r", encoding="utf-8")
        else:
            bonusf = None
        
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
        
        tokdict["$PAD"] = len(toklist)
        toklist.append("$PAD")
        
        self.unkl = ["*UNK*", "*UNKre_num*", "*UNKre_pc*", "*UNKyl*", "*UNKate*", "*UNKated*", "*UNKed*", "*UNKing*", "*UNKic*", "*UNKtion*", "*UNKal*", "*UNKive*", "*UNKly*", "*UNKane*", "*UNKene*", "*UNKyne*", "*UNKine*", "*UNKase*", "*UNKble*", "*UNKity*", "*UNKrgic*", "*UNKein*", "*UNKin*", "*UNKose*", "*UNKone*", "*UNKdehyde*", "*UNKol*", "*UNKide*", "*UNKite*", "*UNKium*",
                     "*UNKs*",
                        #"*UNK-*"
                    #"*UNK_nocd*", "*UNK_noc*", "*UNK_nolc*", "*UNK_no2lc*",
                     #"*UNK_dandc*",
                    ]

        self.unkcache = {}
        
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
        for u in self.unkl:
            tokdict[u] = len(toklist)
            toklist.append(u)
            
        #tokdict["*UNK*"] = len(toklist)
        #toklist.append("*UNK*")
        
        ohn = len(toklist)

        if False:
            # Initialise embeddings using GloVe if present
            em = None
            if glovefile is not None:
                t = time.time()
                ei = {}
                f = open(glovefile, "r", encoding="utf-8")
                for l in f:
                    ll = l.strip().split()
                    #ll = [i.value for i in ChemTokeniser(l.strip(), clm=True).tokens]
                    w = ll[0]
                    c = np.asarray(ll[1:], dtype='float32')
                    ei[w] = c
                em = np.zeros((ohn, 300))
                for i in range(ohn):
                    if toklist[i] in ei: em[i] = ei[toklist[i]]
                print("Embeddings read in:", time.time() - t, file=sys.stderr)

        # Initialise embeddings using GloVe, must always be present
        em = None
        t = time.time()
        ei = {}
        self.ei = ei
        ei["$PAD"] = np.array([0]*300)
        f = open(glovefile, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split()
            w = ll[0]
            c = np.asarray(ll[1:], dtype='float32')
            ei[w] = c
        if "unk" not in ei and "<unk>" in ei: ei["unk"] = ei["<unk>"]
        if "unk" not in ei: ei["unk"] = np.array([0]*300)
        if "*UNK*" not in ei: ei["*UNK*"] = ei["unk"]
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
        fzr = Featurizer(train, extrachem)
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
        
        
        # Bonus
        if bonusf is not None:
            self.bonusf = bonusf

            bonustestsents = []
            print("Start bonus ", datetime.now(), file=sys.stderr)
            nathead = 0
            while(len(bonustestsents) < 1000):
                nathead += 1
                l = bonusf.readline()
                ll = [i.value for i in ChemTokeniser(l.strip(), clm=True).tokens]
                #ll = l.strip().split()
                #print(ll)
                if(len(ll) < 2): continue
                if(len(ll) > 5000): continue
                bonustestsents.append(ll)
            print(datetime.now(), file=sys.stderr)
            bonustestsents = sorted(bonustestsents, key=lambda x:len(x))

        include_xei = False
        if ntype == 0:
            # default, simple
        #if bonusf is None:
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
            bll = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001), dropout=0.5, recurrent_dropout=0.5), merge_mode="concat", name="lstm")(ml)
            blld = bll
            #bll = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001)), merge_mode="concat", name="lstm")(ml)
            #blld = Dropout(0.5)(bll)
            dl = TimeDistributed(Dense(len(lablist), activation="softmax"), name="output")(blld)
            model = Model(inputs=[ei, il], outputs=dl )
            #crms = RMSprop(lr=0.0005)
            #model.compile(loss='categorical_crossentropy', optimizer=crms, metrics=['accuracy'])
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            self.model = model
        elif ntype == 1:
            include_xei = True
            il = Input(shape=(None, self.nilen))
            cl = Conv1D(256, 3, padding='same', activation='relu', input_shape=(None, self.nilen), name="conv")(il)
            cdl = Dropout(0.5)(cl)
            ei = Input(shape=(None,), dtype='int32')
            if em is not None:
                el = Embedding(ohn, 300, weights=[em])(ei)
            else:
                el = Embedding(ohn, 300)(ei)
                
            xei = Input(shape=(None, 300))
            #el = Conv1D(256, 3, padding='same', activation='relu', input_shape=(None, 300), name="el")(ei)
            #el = TimeDistributed(Activation("linear"))(ei)
            xel = TimeDistributed(Dense(128, activation='relu', input_shape=(None, 300)), name="el")(xei)
            xedl = Dropout(0.5)(xel)
            
            ml = concatenate([cdl, el, xedl])
            bll = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001)), merge_mode="concat", name="lstm")(ml)
            blld = Dropout(0.5)(bll)
            dl = TimeDistributed(Dense(len(lablist), activation="softmax"), name="output")(blld)
            model = Model(inputs=[xei, ei, il], outputs=dl)
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            self.model = model
        else:
            include_xei = ntype == 2
            # Set up the keras model
            il = Input(shape=(None, self.nilen))
            cl = Conv1D(256, 3, padding='same', activation='relu', input_shape=(None, self.nilen), name="conv")(il)
            cdl = Dropout(0.5)(cl)

            xei = Input(shape=(None, 300))
            el = TimeDistributed(Dense(128, activation='relu', input_shape=(None, 300)), name="el")(xei)
            edl = Dropout(0.5)(el)

            aei = Input(shape=(None,), dtype='int32')
            aet = Embedding(ohn, 300, weights=[em])
            ael = aet(aei)

            #boll = CuDNNLSTM(300, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001))(ael)
            boll = CuDNNLSTM(300, return_sequences=True)(ael)
            boll = Dropout(0.5)(boll)
            
            #bollr = CuDNNLSTM(300, return_sequences=True, go_backwards=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001))(ael)
            bollr = CuDNNLSTM(300, return_sequences=True, go_backwards=True)(ael)
            bollr = Dropout(0.5)(bollr)
            bollrr = Lambda(lambda xx: K.reverse(xx, 1))(bollr)


            if include_xei:
                ml = concatenate([cdl, edl, boll, bollrr])
            else:
                ml = concatenate([cdl, boll, bollrr])
            #ml = concatenate([cdl, edl, ael])
            bll = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001)), merge_mode="concat", name="lstm")(ml)
            bll = Dropout(0.5)(bll)
            dl = TimeDistributed(Dense(len(lablist), activation="softmax"), name="output")(bll)
            if include_xei:
                model = Model(inputs=[xei, aei, il], outputs=dl)
            else:
                model = Model(inputs=[aei, il], outputs=dl)
            #crms = RMSprop(lr=0.0005)
            #adam = Adam() # default lr: 0.001
            #adam=Adam(lr=0.0005)
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            #model.compile(loss='categorical_crossentropy', optimizer=crms, metrics=['accuracy'])
            self.model = model

            bonusmodeltype = 1
            self.bonusmodeltype = bonusmodeltype
            if bonusmodeltype == 0:        
                bonll = TimeDistributed(Dense(300, activation="relu"))(boll)
                bodl = TimeDistributed(Dense(300))(bonll)
                bonllr = TimeDistributed(Dense(300, activation="relu"))(bollrr)
                bodlr = TimeDistributed(Dense(300))(bonllr)
                bonusmodel = Model(inputs=aei, outputs=[bodl, bodlr])
                bonusmodel.compile(loss='mse', optimizer='rmsprop')
            elif bonusmodeltype == 1:
                bonextil = Input(shape=(None,), dtype='int32')
                boprevil = Input(shape=(None,), dtype='int32')
                bonel = aet(bonextil)
                boncatted = concatenate([boll, bonel])
                boncompl = TimeDistributed(Dense(300, activation='relu'))(boncatted)
                bonoutl = TimeDistributed(Dense(1, activation='sigmoid'))(boncompl)
                bopel = aet(boprevil)
                bopcatted = concatenate([bollrr, bopel])
                bopcompl = TimeDistributed(Dense(300, activation='relu'))(bopcatted)
                bopoutl = TimeDistributed(Dense(1, activation='sigmoid'))(bopcompl)
                bonusmodel = Model(inputs=[aei, bonextil, boprevil], outputs=[bonoutl, bopoutl])
                #bonusmodel.compile(loss='binary_crossentropy', optimizer='rmsprop')
                bonusmodel.compile(loss='binary_crossentropy', optimizer='adam')
 
        
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
        #for epoch in range(1):
        for epoch in range(20):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            
            if bonusf is not None and epoch < 1:
                if bonusmodeltype == 0:
                    for subepoch in range(26):
                        bsize = 32
                        if subepoch > 0:
                            bonussents = []
                            # increase to 30000 or higher once we're done
                            # 60000
                            while(len(bonussents) < (60000*4)):
                            #while(len(bonussents) < 60000):
                                l = bonusf.readline()
                                if l == "":
                                    bonusf = open("/data/raw/patents/catted_pat_ft_shuffle.txt", "r", encoding="utf-8")
                                    self.bonusf = bonusf
                                    # Skip the test head
                                    for i in range(nathead): bonusf.readline()

                                ll = [i.value for i in ChemTokeniser(l.strip(), clm=True).tokens]
                                if(len(ll) < 2): continue
                                if(len(ll) > 5000): continue
                                bonussents.append(ll)
                            losses = []
                            ss = sorted(bonussents, key=lambda x:len(x))
                            for i in range(math.ceil(len(bonussents) / bsize)):
                                batch = ss[i*bsize:min(len(bonussents),(i+1)*bsize)]
                                ml = max([len(i) for i in batch])
                                #print(ml, end=" ")
                                for i in batch:
                                    i += ["$PAD"] * (ml - len(i))

                                x = np.array([[tokdict[i] for i in j] for j in batch])
                                y = np.array([[ei[i] if i in ei else ei["<unk>"] for i in j[1:] + ["$PAD"]] for j in batch])
                                y2 = np.array([[ei[i] if i in ei else ei["<unk>"] for i in ["$PAD"] + j[:-1]] for j in batch])
                                losses.append(bonusmodel.train_on_batch(x, [y, y2]))
                            #print()
                            l1 = [i[1] for i in losses]
                            l2 = [i[2] for i in losses]
                            #print(losses[0])
                            #print(n, sum(losses)/len(losses), datetime.now())
                            #print(n, sum(l1)/len(l1), sum(l2)/len(l2), datetime.now())

                        tlosses = []
                        tsum = 0
                        for i in range(math.ceil(len(bonustestsents) / bsize)):
                            batch = bonustestsents[i*bsize:min(len(bonustestsents),(i+1)*bsize)]
                            ml = max([len(i) for i in batch])
                            lsum = sum([len(i) for i in batch])
                            for i in batch:
                                i += ["$PAD"] * (ml - len(i))

                            x = np.array([[tokdict[i] for i in j] for j in batch])
                            y = np.array([[ei[i] if i in ei else ei["<unk>"] for i in j[1:] + ["$PAD"]] for j in batch])
                            y2 = np.array([[ei[i] if i in ei else ei["<unk>"] for i in ["$PAD"] + j[:-1]] for j in batch])
                            loss = bonusmodel.test_on_batch(x, [y, y2])
                            tlosses.append([i * lsum for i in loss])
                            tsum += lsum
                        ll1 = [i[1] for i in tlosses]
                        ll2 = [i[2] for i in tlosses]
                        if subepoch > 0:
                            print(sum(l1)/len(l1), sum(l2)/len(l2), sum(ll1) / tsum, sum(ll2) / tsum, datetime.now(), file=sys.stderr)
                        else:
                            print("Initial:", sum(ll1) / tsum, sum(ll2) / tsum, file=sys.stderr)
                elif bonusmodeltype == 1:
                    #for subepoch in range(10):
                    for subepoch in range(20):
                        bsize = 32
                        bonussents = []
                        # increase to 30000 or higher once we're done
                        # 60000
                        #while(len(bonussents) < 6000):
                        #while(len(bonussents) < 12000):
                        while(len(bonussents) < mbs):
                            l = bonusf.readline()
                            if l == "":
                                print("Wrap around", file=sys.stderr)
                                bonusf = open(bonusfn, "r", encoding="utf-8")
                                self.bonusf = bonusf
                                # Skip the test head
                                for i in range(nathead): bonusf.readline()
                            ll = [i.value for i in ChemTokeniser(l.strip(), clm=True).tokens]
                            if(len(ll) < 2): continue
                            if(len(ll) > 1000): continue
                            bonussents.append(ll)
                        losses = []
                        catl = []
                        for i in bonussents: catl.extend([self.tokdict[j] if j in self.tokdict else self.tokdict[self.classify_unk(j)] for j in i])
                        ss = sorted(bonussents, key=lambda x:len(x))
                        for i in range(math.ceil(len(bonussents) / bsize)):
                            batch = ss[i*bsize:min(len(bonussents),(i+1)*bsize)]
                            ml = max([len(i) for i in batch])
                            #print(ml, end=" ")
                            x = []
                            xn = []
                            xp = []
                            yn = []
                            yp = []
                            for i in batch:
                                i += ["$PAD"] * (ml - len(i))
                                padn = tokdict["$PAD"]
                                seq = [self.tokdict[j] if j in self.tokdict else self.tokdict[self.classify_unk(j)] for j in i]
                                choice = [random.randint(0,1) for j in i]
                                answer = [[random.choice(catl), seq[j]][choice[j]] for j in range(len(seq))]
                                choice = [[1] if answer[j] == seq[j] else [0] for j in range(len(seq))]
                                x.append(seq)
                                xn.append(answer[1:] + [padn])
                                xp.append([padn] + answer[:-1])
                                yn.append(choice[1:] + [[0]])
                                yp.append([[0]] + choice[:-1])

                            x = np.array(x)
                            xn = np.array(xn)
                            xp = np.array(xp)
                            yn = np.array(yn)
                            yp = np.array(yp)
                            losses.append(bonusmodel.train_on_batch([x, xn, xp], [yn, yp]))
                        l1 = [i[1] for i in losses]
                        l2 = [i[2] for i in losses]

                        print(subepoch, sum(l1)/len(l1), sum(l2)/len(l2), datetime.now(), file=sys.stderr)

                        
                            
            
            self.include_xei = include_xei
            # Train in batches of different sizes - randomize the order of sizes
            random.shuffle(sizes)
            tnt = 0
            for size in sizes: tnt += size * len(train_l_d[size])
            totloss = 0
            totacc = 0
            div = 0    
            for size in sizes:
                if size == 1: continue # For unknown reasons we can't train on a single token
                batch = train_l_d[size]
                
                tx2 = np.array([seq["ni"] for seq in batch])                    
                ty = np.array([[tobits(i, len(lablist)) for i in seq["bion"]] for seq in batch])
                if include_xei: 
                    tx = np.array([seq["worde"] for seq in batch])
                    txsesqui = np.array([seq["wordn"] for seq in batch])
                    history = model.fit([tx, txsesqui, tx2], ty, verbose=0, epochs=1)
                else:
                    tx = np.array([seq["wordn"] for seq in batch])
                    history = model.fit([tx, tx2], ty, verbose=0, epochs=1)
                div += size * len(batch)
                totloss += history.history["loss"][0] * size * len(batch)
                totacc += history.history["acc"][0] * size * len(batch)
                # This trains in mini-batches
                
            print("Trained at", datetime.now(), "Loss", totloss / div, "Acc", totacc / div, file=sys.stderr)
            #print("Trained at", datetime.now(), file=sys.stderr)
            self.model.save("epoch_%s_%s.h5" % (epoch, runname))
            # Evaluate
            tp_all = 0
            fp_all = 0
            fn_all = 0
           # cmplogf = open("cmplog.txt", "w", encoding="utf-8")
            for i in range(len(test)):
                
                enttype = None
                entstart = 0
                ts = test[i]
                ents = [("E", i[2], i[3]) for i in ts["ents"]]
                if include_xei:
                    mm = model.predict([np.array([ts["worde"]]), np.array([ts["wordn"]]), np.array([ts["ni"]])])[0]                    
                else:
                    mm = model.predict([np.array([ts["wordn"]]), np.array([ts["ni"]])])[0]
                #print(i, ts["wordn"], file=cmplogf)
                
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
                #entl = sorted(pents)
                #for j in entl:
                #    print(i, j, file=cmplogf)
                
            f = (2*tp_all/(tp_all+tp_all+fp_all+fn_all))
            print("TP", tp_all, "FP", fp_all, "FN", fn_all, "F", f, "Precision", tp_all/(tp_all+fp_all), "Recall", tp_all/(tp_all+fn_all), file=sys.stderr)
            if f > best_f:
                print("Best so far", file=sys.stderr)
                best_f = f
                best_epoch = epoch
            #print(internal_eval(textfile, annotfile,self))
            #cmplogf.close()
            
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
        self.unkcache = {}
        
        jf = open(jfile, "r", encoding="utf-8")
        jo = json.load(jf)
        jf.close()
        self.tokdict = jo["tokdict"]
        self.lablist = jo["lablist"]
        self.fzr = Featurizer(None, jo["fzr"])
        print("Auxillary information read at", datetime.now(), file=sys.stderr)
        self.model = load_model(mfile)
        print("Traditional Model read at", datetime.now(), file=sys.stderr)
    
    def process(self, instr, threshold=0.5, domonly=True, logf=None):
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
        if len(instr) == 0: return results
        seq = self._str_to_seq(instr)
        if len(seq["tokens"]) == 0: return results
        self._prepare_seqs([seq], False) # Not verbose
        if logf is not None: print(seq["wordn"], file=logf)
        mm = self.model.predict([np.array([seq["wordn"]]), np.array([seq["ni"]])])[0]
        seq["tagfeat"] = mm
        pents, pxe = sobie_scores_to_char_ents(seq, threshold, instr)
        if domonly:
            pents = [i for i in pents if pxe[i]["dom"]]
        for ent in pents:
            results.append((ent[1], ent[2], instr[ent[1]:ent[2]], pxe[ent]["score"], pxe[ent]["dom"]))
        return results
    
    def batchprocess(self, instrs, threshold=0.5, domonly=True):
        pairs = [(n, self._str_to_seq(i)) for n, i in enumerate(instrs)]
        seqs = [i[1] for i in pairs]
        self._prepare_seqs(seqs, False)
        seq_l_d = defaultdict(lambda: [])
        res = [list() for i in instrs]
        for pair in pairs:
            seq = pair[1]
            l = len(seq["tokens"])
            seq_l_d[len(seq["tokens"])].append(pair)
        for l in seq_l_d:
            if l == 0:
                continue
            else:
                ppairs = seq_l_d[l]
                if self.include_xei:
                    mm = self.model.predict([np.array([p[1]["worde"] for p in ppairs]), np.array([p[1]["wordn"] for p in ppairs]), np.array([p[1]["ni"] for p in ppairs])])
                else:
                    mm = self.model.predict([np.array([p[1]["wordn"] for p in ppairs]), np.array([p[1]["ni"] for p in ppairs])])
                for n, p in enumerate(ppairs):
                    p[1]["tagfeat"] = mm[n]
                    pents, pxe = sobie_scores_to_char_ents(p[1], threshold, p[1]["str"])
                    if domonly:
                        pents = [i for i in pents if pxe[i]["dom"]]
                    rr = []
                    for ent in pents:
                        rr.append((ent[1], ent[2], p[1]["str"][ent[1]:ent[2]], pxe[ent]["score"], pxe[ent]["dom"]))
                    res[p[0]] = rr
        return res
