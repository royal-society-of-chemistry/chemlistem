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
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers import Input, CuDNNLSTM, concatenate, GlobalMaxPooling1D, CuDNNGRU
from keras.models import Model, load_model
import keras.backend as K
import keras.regularizers
import numpy as np

from .utils import tobits, sobie_scores_to_char_ents, get_file
from .corpusreader import CorpusReader

charstr = "abcdefghijklmonpqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ0123456789.,-[](){};:'\"^$%=/\\<>@_*+?! "
chard = {charstr[i]:i+1 for i in range(len(charstr))}
charn = len(charstr) + 1

defaultmodel = None
def _getpath(fn):
    if __name__=="__main__": return fn
    d = os.path.split(__file__)[0]
    return os.path.join(d, fn)

def get_mini_model():
    """
    Gets the default pre-trained minimalist model, loading if necessary.
    
    Returns:
        An MiniModel
    """
    global defaultmodel
    if defaultmodel is not None: return defaultmodel
    mm = MiniModel()
    f = get_file("default_minimodel_0.0.1.h5")
    mm.load(f)
    defaultmodel = mm
    return defaultmodel

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

class MiniModelCuDNN(object):
    """
    A "minimalist" model for chemical named entity recognition - works character-by-character, does not use
    rich features, does use multiple bidirectional LSTM layers.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        pass
        
    def make_str_batches(self, items, bsize=32):
        sitems = sorted(items, key=lambda x:len(x))
        batches = to_batches(sitems, bsize)
        nbatches = []
        for batch in batches:
            ml = max(len(i) for i in batch)
            nbatch = []
            for i in batch:
                ldiff = ml - len(i)
                if ldiff > 0:
                    i += " " * ldiff
                nbatch.append(i)
            nbatches.append(nbatch) 
        random.shuffle(nbatches)
        return nbatches
    
    def get_unsup_batches(self, n, bsize, loopround=True):
        lines = []
        while n == 0 or len(lines) < n * bsize:
            line = self.usf.readline()
            if line is None or line == "":
                self.usf.close()
                self.usf = open(self.usfn, "r", encoding="utf-8")
                if not loopround: break
                continue
            line = line.strip()
            if len(line) > 1:
                lines.append(line)
        usb = self.make_str_batches(lines, bsize)
        return usb
        
    def train_unsup_batch(self, batch):
        xl = []
        yfl = []
        ybl = []
        for s in batch:
            cns = [_char_to_num(i) for i in s]
            yy = [tobits(i, charn) for i in cns]
            yyf = yy[1:] + [tobits(0, charn)]
            yyb = [tobits(0, charn)] + yy[:-1]
            xl.append(cns)
            yfl.append(yyf)
            ybl.append(yyb)
        tx = np.array(xl)
        tyf = np.array(yfl)
        tyb = np.array(ybl)
        #print(tx.shape, tyf.shape, tyb.shape)
        res = self.predmodel.train_on_batch(tx, [tyf, tyb])
        return res
    
    def make_dictmodel_batches(self, bsize=32):
        words = set()
        elements = set()
        chebi = set()
        f = open(_getpath("words.txt"), "r", encoding="utf-8", errors="replace")
        for l in f:
            l=l.strip()
            if len(l) > 1:
                words.add(l)
        f = open(_getpath("elements.txt"), "r", encoding="utf-8", errors="replace")
        for l in f:
            l=l.strip()
            if len(l) > 1:
                elements.add(l)
        f = open(_getpath("chebinames.txt"), "r", encoding="utf-8", errors="replace")
        for l in f:
            l=l.strip()
            if len(l) > 1:
                for i in l.split():
                    chebi.add(i)
        wl = words | elements | chebi
        pairs = []
        for i in wl:
            pairs.append([i, (1 if i in words else 0, 1 if i in elements else 0, 1 if i in chebi else 0)])
        pairs = sorted(pairs, key=lambda x:len(x[0]))
        batches = to_batches(pairs, 32)
        nbatches = []
        for batch in batches:
            ml = max(len(i[0]) for i in batch)
            nbatch = []
            for i in batch:
                ldiff = ml - len(i[0])
                if ldiff > 0:
                    i[0] += " " * ldiff
                nbatch.append(i)
            nbatches.append(nbatch) 
        random.shuffle(nbatches)
        self.dictbatches = nbatches
        return nbatches
    
    def train_dictmodel_batch(self, batch):
        xl = []
        yl = []
        for item in batch:
            cns = [_char_to_num(i) for i in item[0]]
            yy = item[1]
            xl.append(cns)
            yl.append(yy)
        tx = np.array(xl)
        ty = np.array(yl)
        #print(tx.shape, tyf.shape, tyb.shape)
        res = self.dictmodel.train_on_batch(tx, ty)
        return res
    
    
    def train_dictmodel(self):
        print(len(self.dictbatches))
        rl = []
        for n, b in enumerate(self.dictbatches):
            res = self.train_dictmodel_batch(b)
            rl.append(res)
            if n % 100 == 0: 
                print(n, np.mean(rl), res, datetime.now())
                rl = []
            
    def train(self, textfile, annotfile, runname, unsupfile, nunsup=0, unsupfile2=None):
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
        print("Make train dict at", datetime.now(), file=sys.stderr)
    
        train_l_d = {}
        for seq in train:
            l = len(seq["tokens"])
            if l not in train_l_d: train_l_d[l] = []
            train_l_d[l].append(seq)
        sizes = list(train_l_d.keys())
        
        test_l_d = {}
        for seq in test:
            l = len(seq["tokens"])
            if l not in test_l_d: test_l_d[l] = []
            test_l_d[l].append(seq)
        test_sizes = list(test_l_d.keys())
    
        # Set up the keras model
        il = Input(shape=(None, ), dtype='int32')
        el = Embedding(charn, 200, name="embed")(il)
        
        rnn = CuDNNGRU
        #rnn = CuDNNLSTM
        rnn2 = CuDNNLSTM
        
        l1f = rnn(128, return_sequences=True, name="lstmf")(el) # was 128
        l1b = rnn(128, return_sequences=True, go_backwards=True, name="lstmr")(el) # was 128
        l1br = Lambda(lambda xx: K.reverse(xx, 1))(l1b)
        bl1 = concatenate([l1f, l1br])
        
        #bl1 = Bidirectional(rnn(128, return_sequences=True), merge_mode="concat", name="lstm1")(el)
        bl1d = Dropout(0.5)(bl1)
        bl2 = Bidirectional(rnn2(64, return_sequences=True), merge_mode="concat", name="lstm2")(bl1d)
        bl2d = Dropout(0.5)(bl2)
        bl3 = Bidirectional(rnn2(64, return_sequences=True), merge_mode="concat", name="lstm3")(bl2d)
        bl3d = Dropout(0.5)(bl3)

        dl = TimeDistributed(Dense(len(self.lablist), activation="softmax"), name="output")(bl3d)
        model = Model(inputs=il, outputs=dl)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.model = model
        
        #l1fd = Dropout(0.5)(l1f)
        #l1brd = Dropout(0.5)(l1br)
        dl_f = TimeDistributed(Dense(charn, activation="softmax"), name="pred_fwd")(l1f)
        dl_b = TimeDistributed(Dense(charn, activation="softmax"), name="pred_back2")(l1br)
        predmodel = Model(inputs=il, outputs = [dl_f, dl_b])
        predmodel.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.predmodel = predmodel
        
        gmp = GlobalMaxPooling1D()(bl3d) # was briefly bl2d but that wasn't so good
        cxd = Dense(3, activation="sigmoid", name="chemout")(gmp)
        dictmodel = Model(inputs=il, outputs = cxd)
        dictmodel.compile(loss='binary_crossentropy', optimizer='rmsprop')
        self.dictmodel = dictmodel
        
        self.make_dictmodel_batches()
    
        self.usfn = unsupfile
        self.usf = open(self.usfn, "r", encoding="utf-8")
        #self.usl = []
        #for l in self.usf:
        #    l = l.strip()
        #    if len(l) > 1:
        #        self.usl.append(l)
        #self.usb = self.make_str_batches(self.usl, 32)

        
        if nunsup != 0:
            if nunsup == -1: nunsup = 0
            self.usb = self.get_unsup_batches(nunsup, 32, loopround=False)
            print(len(self.usb), "batches of unsupervised")

            rl = []
            maxl = 0
            for n, b in enumerate(self.usb):
                #if len(b[0]) > maxl:
                #    maxl = len(b[0])
                #    print("New max", maxl)
                #if n < 4000: continue
                if len(b[0]) > 16384:
                    print("Skipped batch, length", len(b[0]))
                    continue
                res = self.train_unsup_batch(b)
                rl.append(res[0])
                if n % 100 == 0: 
                    print(n, np.mean(rl), res, datetime.now())
                    rl = []

        if unsupfile2 is not None:
            self.usfn = unsupfile2
            self.usf = open(self.usfn, "r", encoding="utf-8")
            self.usb = self.get_unsup_batches(0, 32, loopround=False)
            print(len(self.usb), "batches of second unsupervised")

            rl = []
            maxl = 0
            for n, b in enumerate(self.usb):
                #if len(b[0]) > maxl:
                #    maxl = len(b[0])
                #    print("New max", maxl)
                #if n < 4000: continue
                if len(b[0]) > 16384:
                    print("Skipped batch, length", len(b[0]))
                    continue
                res = self.train_unsup_batch(b)
                rl.append(res[0])
                if n % 100 == 0: 
                    print(n, np.mean(rl), res, datetime.now())
                    rl = []
                    
        best_epoch = -1
        best_f = 0.0
    
        # OK, start actually training
        for epoch in range(30):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            # Train in batches of different sizes - randomize the order of sizes
            # Except for the first few epochs - train on the smallest examples first
            if epoch > 4: random.shuffle(sizes) # For unknown reasons we can't train on a single token (i.e. character)
            for size in sizes:
                if size == 1: continue
                if len(self.dictbatches) > 0:
                    if len(self.dictbatches) == 1: print("Last Dict", datetime.now())
                    self.train_dictmodel_batch(self.dictbatches.pop())
                batch = train_l_d[size]
                tx = np.array([seq["wordn"] for seq in batch])
                ty = np.array([[tobits(i, len(self.lablist)) for i in seq["bion"]] for seq in batch])
                # This trains in mini-batches
                model.fit(tx, ty, verbose=0, epochs=1)
            print("Trained at", datetime.now(), file=sys.stderr)
            model.save("epoch_%s_%s.h5" % (epoch, runname))
            # Evaluate
            tp_all = 0
            fp_all = 0
            fn_all = 0
            for size in test_sizes:
                batch = test_l_d[size] 
                tx = np.array([seq["wordn"] for seq in batch])
                mmb = model.predict(tx)
            #for i in range(len(test)):
                for ii in range(len(batch)):
                    enttype = None
                    entstart = 0
                    ts = batch[ii]
                    ents = [("E", i[2], i[3]) for i in ts["ents"]]
                    mm = mmb[ii]
                    #mm = model.predict([np.array([ts["wordn"]])])[0]

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

