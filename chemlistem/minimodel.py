import time
import io
import sys
import os
import random
import re
import json
import shutil
from collections import defaultdict
from datetime import datetime
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers import Input, concatenate, GlobalMaxPooling1D
try:
    from keras.layers import CuDNNLSTM
except:
    CuDNNLSTM = None
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
defaultmodelgpu = False
def _getpath(fn):
    if __name__=="__main__": return fn
    d = os.path.split(__file__)[0]
    return os.path.join(d, fn)

def get_mini_model(version="0.1.0", gpu=False):
    """
    Gets the default pre-trained minimalist model, loading if necessary.
    
    Args:
        version: the version number on BitBucket.
        gpu: whether to use CuDNNLSTM.
    
    Returns:
        A MiniModel
    """
    global defaultmodel, defaultmodelgpu
    if defaultmodel is not None and gpu == defaultmodelgpu: return defaultmodel
    defaultmodelgpu = gpu

    mm = MiniModel()
    gpustr = "_gpu" if gpu else ""
    f = get_file("default_minimodel%s_%s.h5" % (gpustr, version))
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

class MiniModel(object):
    """
    A "minimalist" model for chemical named entity recognition - works character-by-character, does not use
    rich features, does use multiple bidirectional LSTM layers.
    """

    def __init__(self):
        """
        Empty constructor - use train or load to populate this.
        """
        pass
        
    def _make_str_batches(self, items, bsize=32):
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
    
    def _get_unsup_batches(self, n, bsize, loopround=True):
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
        usb = self._make_str_batches(lines, bsize)
        return usb
        
    def _train_unsup_batch(self, batch):
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
    
    def _train_autoenc_batch(self, batch):
        xl = []
        yl = []
        for s in batch:
            cns = [_char_to_num(i) for i in s]
            yy = [tobits(i, charn) for i in cns]
            for i in range(len(s)):
                if random.random() < 0.2:
                    cns[i] = 0
                    #cns[i] = random.randint(0,charn-1)
            xl.append(cns)
            yl.append(yy)
        tx = np.array(xl)
        ty = np.array(yl)
        #print(tx.shape, tyf.shape, tyb.shape)
        res = self.autoencmodel.train_on_batch(tx, ty)
        return res
        
    
    def _make_dictmodel_batches(self, bsize=32):
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
    
    def _train_dictmodel_batch(self, batch):
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
    
    
    def _train_dictmodel(self):
        print(len(self.dictbatches))
        rl = []
        for n, b in enumerate(self.dictbatches):
            res = self.train_dictmodel_batch(b)
            rl.append(res)
            if n % 100 == 0: 
                print(n, np.mean(rl), res, datetime.now())
                rl = []

    def _make_model(self, gpu):
        il = Input(shape=(None, ), dtype='int32')
        el = Embedding(charn, 200, name="embed")(il)
        
        if gpu:
            l1f = CuDNNLSTM(128, return_sequences=True, name="lstmf")(el) # was 128
            l1b = CuDNNLSTM(128, return_sequences=True, go_backwards=True, name="lstmr")(el) # was 128
            l1br = Lambda(lambda xx: K.reverse(xx, 1))(l1b)
            bl1 = concatenate([l1f, l1br])

            bl1d = Dropout(0.5)(bl1)
            bl2 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode="concat", name="lstm2")(bl1d)
            bl2d = Dropout(0.5)(bl2)
            bl3 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode="concat", name="lstm3")(bl2d)
            bl3d = Dropout(0.5)(bl3)
        else:
            l1f = LSTM(128, recurrent_activation="sigmoid", dropout=0.5, recurrent_dropout=0.5, return_sequences=True, name="lstmf")(el) # was 128
            l1b = LSTM(128, recurrent_activation="sigmoid", dropout=0.5, recurrent_dropout=0.5, return_sequences=True, go_backwards=True, name="lstmr")(el) # was 128
            l1br = Lambda(lambda xx: K.reverse(xx, 1))(l1b)
            bl1 = concatenate([l1f, l1br])

            bl1d = bl1
            bl2 = Bidirectional(LSTM(64, recurrent_activation="sigmoid", dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode="concat", name="lstm2")(bl1d)
            bl2d = bl2
            bl3 = Bidirectional(LSTM(64, recurrent_activation="sigmoid", dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode="concat", name="lstm3")(bl2d)
            bl3d = bl3

        dl = TimeDistributed(Dense(len(self.lablist), activation="softmax"), name="output")(bl3d)
        model = Model(inputs=il, outputs=dl)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model, il, l1f, l1br, bl3d
                
    def train(self, textfile, annotfile, runname, unsupfile=None, nunsup=-1, unsupcfg={"pre_pred", "inter_dict"}, gpu=False):
        """
        Train a new MiniModel.
                
        This produces one important file:
        
        minimodel_$RUNNAME.h5 - the keras model itself
        
        These consititute the trained model.
        
        It also produces several files named:
        
        epoch_$EPOCHNUM_$RUNAME.h5
        
        These are the keras models for each epoch (the auxilliary information doesn't change).
        
        Unsupcfg options - this should be a set containing at most one from each of these two groups
        
            First group
            "pre_pred" - do predictive pretraining at the beginning
            "inter_pred" - interleave predictive pretraining with regular training
            
            "pre_pre_dict" - do dictionary pretraining at the beginning, before predictive
            "pre_dict" - do dictionary training before regular training, but after predictive
            "inter_pre_dict" - interleave dictionary training and predictive training
            "inter_dict" - interleave dictionary training with regular training
            
        Example: {"pre_pred", "inter_dict"} gives the best results
        
        Args:
            textfile: the filename of the file containing the BioCreative training text - e.g. "BioCreative V.5 training set.txt"
            annotfile: the filename of the containing the BioCreative training annotations - e.g. "CEMP_BioCreative V.5 training set annot.tsv"
            runname: a string, part of the output filenames.
            upsupfile: file of lines of example text for unsupervised learning
            nunsup: how many lines to use from the file. 0 is no unsupervised learning. -1 is all the lines, once only.
            unsupcfg: None, or a set of strings - see above.
            gpu: True if you want to use CUDNNLSTM - this is faster but needs a GPU and is slightly less accurate. Otherwise False.
        """    
        if unsupcfg is None: unsupcfg = set()
        
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
        model, il, l1f, l1br, bl3d = self._make_model(gpu)
        
        self.model = model
        
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
        
        self._make_dictmodel_batches()
    
        self.usfn = unsupfile
        if unsupfile is not None:
            self.usf = open(self.usfn, "r", encoding="utf-8")

            self.usb = []
            if "pre_pre_dict" in unsupcfg:
                while len(self.dictbatches) > 0:
                    self._train_dictmodel_batch(self.dictbatches.pop())

            if "inter_pred" in unsupcfg:
                if nunsup == -1: nunsup = 0
                self.usb = self._get_unsup_batches(nunsup, 32, loopround=False)
                print(len(self.usb), "batches of unsupervised")            

            if nunsup != 0 and "pre_pred" in unsupcfg:
                if nunsup == -1: nunsup = 0
                self.usb = self._get_unsup_batches(nunsup, 32, loopround=False)
                print(len(self.usb), "batches of unsupervised")

                rl = []
                arl = []
                maxl = 0
                for n, b in enumerate(self.usb):
                    if len(b[0]) > 16384:
                        print("Skipped batch, length", len(b[0]))
                        continue
                    if False:
                        res = self._train_autoenc_batch(b)
                        rl.append(res[0])
                        arl.append(res[1])
                    else:
                        res = self._train_unsup_batch(b)
                        rl.append(res[0])
                    if "inter_pre_dict" in unsupcfg and len(self.dictbatches) > 0:
                        if len(self.dictbatches) == 1: print("Last Dict", datetime.now())
                        self._train_dictmodel_batch(self.dictbatches.pop())
                    if n % 100 == 0:
                        if len(arl) > 0:
                            print(n, np.mean(rl), np.mean(arl), res, datetime.now())                       
                            rl = []
                            arl = []
                        else:
                            print(n, np.mean(rl), res, datetime.now())
                            rl = []

            if "pre_dict" in unsupcfg or "inter_pre_dict" in unsupcfg:
                print("Train", len(self.dictbatches), "Dict at", datetime.now())
                while len(self.dictbatches) > 0:
                    self._train_dictmodel_batch(self.dictbatches.pop())
                print("Dict trained at", datetime.now())
                                            
        best_epoch = -1
        best_f = 0.0
    
        # OK, start actually training
        for epoch in range(30):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            # Train in batches of different sizes - randomize the order of sizes
            # Except for the first few epochs - train on (roughly) the smallest examples first
            if epoch > 4: random.shuffle(sizes) 
            for size in sizes:
                if size == 1: continue # For unknown reasons we can't train on a single token (i.e. character)
                if "inter_pred" in unsupcfg and len(self.usb) > 0:
                    if len(self.usb) == 1: print("Last Unsup", datetime.now())
                    self._train_unsup_batch(self.usb.pop())
                if "inter_dict" in unsupcfg and len(self.dictbatches) > 0:
                    if len(self.dictbatches) == 1: print("Last Dict", datetime.now())
                    self._train_dictmodel_batch(self.dictbatches.pop())
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

    def convert_to_no_gpu(self, mfile, mfile_out):
        """
        Load in model data from a model trained with CuDNNLSTMs, convert to LSTM, save a modified version.
        
        Args:
            mfile: the filename of the .h5 file to read
            mfile_out: the filename of the .h5 file to write
        """
        self.lablist = ['S-E', 'O', 'I-E', 'E-E', 'B-E']
        self.labdict = {'O': 1, 'E-E': 3, 'B-E': 4, 'S-E': 0, 'I-E': 2}
        
        model = self._make_model(False)[0]
        model.load_weights(mfile)
        model.save(mfile_out)
        
        self.model = model
        print("Minimalist Model converted at", datetime.now(), file=sys.stderr)
        
        
    def process(self, instr, threshold=0.5, domonly=True):
        """
        Find named entities in a string.
        
        Entities are returned as tuples:
        (start_charater_position, end_character_position, string, score, is_dominant)
        
        Entities are dominant if they are not partially or wholly overlapping with a higher-scoring entity.
        
        Args:
            instr: the string to find entities in.
            threshold: the minimum score for entities.
            domonly: if True, discard non-dominant entities.
        """
        results = []
        if len(instr) == 0: return results
        seq = {}
        seq["tokens"] = list(instr)
        seq["ss"] = instr
        seq["tokstart"] = [i for i in range(len(instr))]
        seq["tokend"] = [i+1 for i in range(len(instr))]
        seq["wordn"] = [_char_to_num(i) for i in seq["tokens"]]
        mm = self.model.predict([np.array([seq["wordn"]])])[0]
        seq["tagfeat"] = mm
        pents, pxe = sobie_scores_to_char_ents(seq, threshold, instr)
        if domonly:
            pents = [i for i in pents if pxe[i]["dom"]]
        for ent in pents:
            results.append((ent[1], ent[2], instr[ent[1]:ent[2]], pxe[ent]["score"], pxe[ent]["dom"]))
        return results
    
    def _str_to_seq(self, s):
        seq = {}
        seq["tokens"] = list(s)
        seq["str"] = s
        seq["tokstart"] = [i for i in range(len(s))]
        seq["tokend"] = [i+1 for i in range(len(s))]
        seq["wordn"] = [_char_to_num(i) for i in seq["tokens"]]
        return seq
    
    def batchprocess(self, instrs, threshold=0.5, domonly=True):
        """
        Find named entities in a set of strings. This is potentially faster as neural network calculations
        run faster in batches.
        
        Entities are returned as tuples:
        (start_charater_position, end_character_position, string, score, is_dominant)
        
        Entities are dominant if they are not partially or wholly overlapping with a higher-scoring entity.
        
        Args:
            instrs: the string to find entities in.
            threshold: the minimum score for entities.
            domonly: if True, discard non-dominant entities.
        """
        pairs = [(n, self._str_to_seq(i)) for n, i in enumerate(instrs)]
        seqs = [i[1] for i in pairs]        
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
                mm = self.model.predict([np.array([p[1]["wordn"] for p in ppairs])])
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

