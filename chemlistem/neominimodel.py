import time
import io
import sys
import os
import random
import re
import json
import shutil
from datetime import datetime
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import Input, CuDNNLSTM
from keras.models import Model, load_model
import keras.regularizers
import numpy as np

from .utils import tobits, sobie_scores_to_char_ents, get_file
from .corpusreader import CorpusReader

charstr = "abcdefghijklmonpqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ0123456789.,-[](){};:'\"^$%=/\\<>@_*+?! "
chard = {charstr[i]:i+1 for i in range(len(charstr))}
charn = len(charstr) + 1

#defaultmodel = None
#
#def get_mini_model():
#    """
#    Gets the default pre-trained minimalist model, loading if necessary.
#    
#    Returns:
#        An MiniModel
#    """
#    global defaultmodel
#    if defaultmodel is not None: return defaultmodel
#    mm = MiniModel()
#    f = get_file("default_minimodel_0.0.1.h5")
#    mm.load(f)
#    defaultmodel = mm
#    return defaultmodel

def _char_to_num(c):
    if c in chard: return chard[c]
    return 0

def to_batches(l, size):
    batches = []
    batch = []
    batches.append(batch)
    for i in l:
        batch.append(i)
        if len(batch) == size:
            batch = []
            batches.append(batch)
    if len(batch) == 0: batches.pop()
    return batches

class NeoMiniModel(object):
    """
    A "minimalist" model for chemical named entity recognition - works character-by-character, does not use
    rich features, does use multiple bidirectional LSTM layers.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        pass
    
    def make_batches(self, items, bsize=32):
        sitems = sorted(items, key=lambda x:len(x["tokens"]))
        batches = to_batches(sitems, bsize)
        for batch in batches:
            ml = max(len(i["tokens"]) for i in batch)
            for i in batch:
                ldiff = ml - len(i["tokens"])
                if ldiff > 0:
                    i["wordn"] += [0] * ldiff
                    i["bion"] += [0] * ldiff
        random.shuffle(batches)
        return batches
                
    
    def train(self, textfile, annotfile, runname):
        """
        Train a new MiniModel.
                
        This produces one important file:
        
        minimodel_$RUNNAME.h5 - the keras model itself
        
        These consititute the trained model.
        
        It also produces several files named:
        
        epoch_$EPOCHNUM_$RUNAME.h5
        
        These are the keras models for each epoch (the auxilliary information doesn't change).
        
        Args:
            textfile: the filename of the file containing the BioCreative training text - e.g. "BioCreative V.5 training set.txt"
            annotfile: the filename of the containing the BioCreative training annotations - e.g. "CEMP_BioCreative V.5 training set annot.tsv"
            runname: a string, part of the output filenames.
        """    
        # Get training and test sequences
        cr = CorpusReader(textfile, annotfile, charbychar=True)
        train = cr.trainseqs
        test = cr.testseqs

        seqs = train+test
        # "wordn" should be "charn" but we're using names lifted from the tradmodel.
        # Anyway, characters to integers to use with embeddings
        for seq in seqs:
            seq["wordn"] = [_char_to_num(i) for i in seq["tokens"]]
        
        self.lablist = ['S-E', 'O', 'I-E', 'E-E', 'B-E']
        self.labdict = {'O': 1, 'E-E': 3, 'B-E': 4, 'S-E': 0, 'I-E': 2}
        
        # convert SOBIE tags to numbers
        for seq in seqs:
            seq["bion"] = [self.labdict[i] for i in seq["bio"]]

        # Gather together sequences by length
        #print("Make train dict at", datetime.now(), file=sys.stderr)
    
        #train_l_d = {}
        #for seq in train:
        #    l = len(seq["tokens"])
        #    if l not in train_l_d: train_l_d[l] = []
        #    train_l_d[l].append(seq)
        #sizes = list(train_l_d.keys())
        print("Make batches at", datetime.now(), file=sys.stderr)
        bsize = 32
        trainbatches = self.make_batches([i for i in train if len(i["tokens"]) > 1], bsize)
        testbatches = self.make_batches([i for i in test if len(i["tokens"]) > 1], bsize)
    
        # Set up the keras model
        il = Input(shape=(None, ), dtype='int32')
        el = Embedding(charn, 200, name="embed")(il)
        bl1 = Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode="concat", name="lstm1")(el)
        bl1d = Dropout(0.5)(bl1)
        bl2 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode="concat", name="lstm2")(bl1d)
        bl2d = Dropout(0.5)(bl2)
        bl3 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode="concat", name="lstm3")(bl2d)
        bl3d = Dropout(0.5)(bl3)
        dl = TimeDistributed(Dense(len(self.lablist), activation="softmax"), name="output")(bl3d)
        model = Model(inputs=il, outputs=dl)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.model = model
        
        best_epoch = -1
        best_f = 0.0
    
        # OK, start actually training
        for epoch in range(30):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            # Train in batches of different sizes - randomize the order of sizes
            # Except for the first few epochs - train on the smallest examples first
            #if epoch > 4: random.shuffle(sizes) # For unknown reasons we can't train on a single token (i.e. character)
            #print(len(trainbatches))
            for n, batch in enumerate(trainbatches):
                #if n % 200 == 0: print(n, datetime.now())
                tx = np.array([seq["wordn"] for seq in batch])
                ty = np.array([[tobits(i, len(self.lablist)) for i in seq["bion"]] for seq in batch])
                model.train_on_batch(tx, ty)
                # This trains in mini-batches
                #model.fit(tx, ty, verbose=0, epochs=1)
            #for size in sizes:
            #    if size == 1: continue
            #    print(size, datetime.now())
            #    batch = train_l_d[size]
            #    tx = np.array([seq["wordn"] for seq in batch])
            #    ty = np.array([[tobits(i, len(self.lablist)) for i in seq["bion"]] for seq in batch])
            #    # This trains in mini-batches
            #    model.fit(tx, ty, verbose=0, epochs=1)
            print("Trained at", datetime.now(), file=sys.stderr)
            model.save("epoch_%s_%s.h5" % (epoch, runname))
            # Evaluate
            tp_all = 0
            fp_all = 0
            fn_all = 0
            for n, batch in enumerate(testbatches):
                #if n % 200 == 0: print(n, datetime.now())
                tx = np.array([seq["wordn"] for seq in batch])
                #mmb = model.predict(tx)
                mmb = model.predict_on_batch(tx)
                #print(mmb.shape)
                for i in range(len(batch)):

                    enttype = None
                    entstart = 0
                    ts = batch[i]
                    ents = [("E", i[2], i[3]) for i in ts["ents"]]
                    mm = mmb[i]
                    if len(mm) > len(ts["tokens"]): mm = mm[:len(ts["tokens"])]

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
            shutil.copyfile("epoch_%s_%s.h5" % (best_epoch, runname), "minimodel_%s.h5" % runname)

    def load(self, mfile):
        """
        Load in model data.
        
        Args:
            mfile: the filename of the .h5 file
        """    
        self.lablist = ['S-E', 'O', 'I-E', 'E-E', 'B-E']
        self.labdict = {'O': 1, 'E-E': 3, 'B-E': 4, 'S-E': 0, 'I-E': 2}
        self.model = load_model(mfile)
        print("Minimalist Model read at", datetime.now(), file=sys.stderr)
        
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
        seq = {}
        seq["tokens"] = list(str)
        seq["ss"] = str
        seq["tokstart"] = [i for i in range(len(str))]
        seq["tokend"] = [i+1 for i in range(len(str))]
        seq["wordn"] = [_char_to_num(i) for i in seq["tokens"]]
        mm = self.model.predict([np.array([seq["wordn"]])])[0]
        seq["tagfeat"] = mm
        pents, pxe = sobie_scores_to_char_ents(seq, threshold, str)
        if domonly:
            pents = [i for i in pents if pxe[i]["dom"]]
        for ent in pents:
            results.append((ent[1], ent[2], str[ent[1]:ent[2]], pxe[ent]["score"], pxe[ent]["dom"]))
        return results

