# Code for our entry in BioCreative VI Task 5. Currently this is here in case the
# description in the paper is unclear, however it is conceivable that this could be
# re-worked into a useable system.
#
# Note - must be run in a directory containing the subdirectories chemprot_test, chemprot_development and chemprot_training,
# containing the contents of the .zip files for the corpora for this challenge.
#
# To use:
# import model
# glovefile = "/data/raw/patents/GloVe/vectors_combined_ft.txt"
# bonusfilename = "/data/raw/medline/medline_catted_shuffle.txt"
# m = model.CPModel("runname", glovefile)
#
#
#

import sys
import math
import time
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import ttest_ind

from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import Input, concatenate
from keras.models import Model, load_model
from keras.optimizers import RMSprop, TFOptimizer
import tensorflow as tf
from keras.regularizers import l1_l2
import keras.backend as K

import corpus

sub = True
sidec = True

def cohend(a, b):
    divisor = math.sqrt((np.std(a)**2+np.std(b)**2)/2)
    if divisor == 0: divisor = 1e-10
    return (np.mean(a)-np.mean(b))/(divisor)

class Instance(object):
    """SFK, Deadmines, RFD, RFK, RFC, Stocks etc."""
    def __init__(self, a, relnn, model, binary=True):
        self.a = a
        self.numtoks = list(a.numtoks)
        self.toktyp = list(a.toktyp)
        self.relnx = a.relnx[relnn,:,:]
        self.relny = a.relny[relnn,:]
        self.binary = binary
        self.e1 = a.gsrelns[relnn].e1
        self.e2 = a.gsrelns[relnn].e2
        
        if binary:
            self.y = [0,1] if self.relny[0] == 0 else [1,0]
        else:
            self.y = self.relny
        self.subtok = list(self.numtoks)
        if sub:        
            for k in range(len(self.numtoks)):
                if self.relnx[k,0] == 1:
                    if self.relnx[k,1] == 1:
                        self.subtok[k] = model.tokdict["$ENTBOTH"]
                    else:
                        self.subtok[k] = model.tokdict["$ENT1"]
                elif self.relnx[k,1] == 1:
                    self.subtok[k] = model.tokdict["$ENT2"]    

    def snippetise(self):
        startpos = None
        endpos = None
        for k in range(len(self.numtoks)):
            if self.relnx[k,0] == 1 or self.relnx[k,1] == 1:
                if startpos is None: startpos = k
                endpos = k
        nstart = max(0, startpos-5)
        nend = min(endpos+6, len(self.numtoks))
        self.todiscard = nend - nstart > 60 
        #print(type(self.numtoks), type(self.toktyp), type(self.relnx))
        self.subtok = self.subtok[nstart:nend]
        self.numtoks = self.numtoks[nstart:nend]
        self.toktyp = self.toktyp[nstart:nend]
        self.relnx = self.relnx[nstart:nend,:]
        #print(type(self.numtoks), type(self.toktyp), type(self.relnx))
            
    def __len__(self):
        return len(self.numtoks)

    def padto(self, mlen):
        initl = len(self)
        if mlen > initl:
            relnx = list(self.relnx)
            for i in range(initl, mlen):
                self.subtok.append(0)
                self.toktyp.append([0]*len(self.toktyp[0]))
                relnx.append([0]*len(relnx[0]))
            self.relnx = np.array(relnx)
        # Other stuff later
    
class Batch(object):
    def __init__(self, instances):
        self.instances = instances
        self.mlen = max([len(i) for i in instances])
        for i in instances:
            i.padto(self.mlen)
        
        self.X = np.array([i.subtok for i in instances])
        self.X1 = np.array([i.toktyp for i in instances])
        self.X2 = np.array([i.relnx for i in instances])
        #print(self.X.shape)
        self.Y = np.array([i.y for i in instances])
    
class CPModel(object):
    
    def make_batches(self, abstracts, discardlong = False, bsize=32):
        instances = []
        truetot = 0
        falsetot = 0
        truethrough = 0
        falsethrough = 0
        for i in abstracts:
            for j in range(len(i.gsrelns)):
                ii = Instance(i, j, self, False)
                ii.snippetise()
                if ii.y[1] == 1:
                    truetot += 1
                else:
                    falsetot += 1
                if (not discardlong) or (not ii.todiscard):
                    instances.append(ii)
                    if ii.y[1] == 1:
                        truethrough += 1
                    else:
                        falsethrough += 1
        print("truetot", truetot, "falsetot", falsetot, "truethrough", truethrough, "falsethrough", falsethrough)
        random.shuffle(instances)
        instances = sorted(instances, key=lambda x:len(x))
        bnum = math.ceil(len(instances)/bsize)
        #batches = [Batch(instances[i::bnum]) for i in range(bnum)]
        batches = [Batch(instances[i*bsize:(i+1)*bsize]) for i in range(bnum)]
        random.shuffle(batches)
        return batches

    
    # runname = a name for saving files for submission. Can be None to not generate these.
    # glovefile = the file containing pre-compiled 300-dimensional vectors
    # bonusfilename = medline abstract contents, one para per line, whitespace-separated tokens
    # do_transfer = boolean for transfer learning
    # use_glove = whether or not to use the pretrained embeddings
    def __init__(self, runname, glovefile, bonusfilename, do_transfer=True, use_glove=True):
        print("Reading corpus at", datetime.now())
        self.corpus = corpus.Corpus()
    
        toklist = []
        tokdict = {}
        tokcounts = Counter()
        self.toklist = toklist
        self.tokdict = tokdict
        self.tokcounts = tokcounts
        
        toklist.append("$PAD")
        toklist.append("$UNK")
        toklist.append("$ENT1")
        toklist.append("$ENT2")
        toklist.append("$ENTBOTH")
        for a in self.corpus.train:
            tokcounts.update([t.value for t in a.tokr.tokens])
        for i, n in tokcounts.most_common():
            if n > 2: toklist.append(i)
        tokdict = defaultdict(lambda: 1)
        tokdict.update({j:i for i, j in enumerate(toklist)})
        self.tokdict = tokdict
        #print(tokdict)
        
        # Initialise embeddings using GloVe, must always be present
        
        em = None
        t = time.time()
        ei = {}
        self.ei = ei
        ei["$PAD"] = np.array([0]*300)
        print("Reading embeddings at", datetime.now(), file=sys.stderr)
        f = open(glovefile, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split()
            w = ll[0]
            c = np.asarray(ll[1:], dtype='float32')
            ei[w] = c
            
            # pad out
            if w not in tokdict and len(toklist) < 50000:
                tokdict[w] = len(toklist)
                toklist.append(w)
            
        if "<unk>" not in ei: ei["<unk>"] = np.array([0]*300)
        if "*UNK*" not in ei: ei["*UNK*"] = ei["<unk>"]
        em = np.zeros((len(toklist), 300))
        for i in range(len(toklist)):
            if toklist[i] in ei: em[i] = ei[toklist[i]]
        print("Embeddings read in:", time.time() - t, file=sys.stderr)
        
        #noutclass = 2
        noutclass = 6
        
        
        for a in self.corpus.train:
            a.numtoks = [tokdict[i.value] for i in a.tokr.tokens]
        for a in self.corpus.test:
            a.numtoks = [tokdict[i.value] for i in a.tokr.tokens]    
        for a in self.corpus.eval:
            a.numtoks = [tokdict[i.value] for i in a.tokr.tokens]    
        
        tokil = Input(shape=(None,), dtype='int32')
        #emb_tensor = Embedding(len(toklist), 300, weights=[em], mask_zero=True)
        if use_glove:
            emb_tensor = Embedding(len(toklist), 300, weights=[em])
        else:
            emb_tensor = Embedding(len(toklist), 300)
            
        embl = emb_tensor(tokil)
        
        
        
        
        #catl = concatenate([embl, ental, relental])
        fwd_ll = LSTM(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(embl)
        back_ll = LSTM(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, go_backwards=True)(embl)
        back_ll_r = Lambda(lambda xx: K.reverse(xx, 1))(back_ll)

        if sidec:
            entil = Input(shape=(None, 16))
            #ental = Activation("linear")(entil)
            ental = Conv1D(48, 3, padding='same', activation='relu', name="conv")(entil)

            relentil = Input(shape=(None, 2))
            #relental = Activation("linear")(relentil)
            relental = Conv1D(6, 3, padding='same', activation='relu', name="conv2")(relentil)

            bll = concatenate([fwd_ll, back_ll_r, ental, relental])
            #bll = concatenate([fwd_ll, back_ll_r, ental, relental, embl])
        else:
            bll = concatenate([fwd_ll, back_ll_r])
        #bll = Bidirectional(LSTM(4, return_sequences=True), merge_mode="concat")(embl)
        #bll = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode="concat")(embl)
        bll2 = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode="concat")(bll)
        gmp1d = GlobalMaxPooling1D()(bll2)
        #ll = LSTM(200)(bll)
        #ll = LSTM(64, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l1_l2(l1=0.000001, l2=0.000001))(bll)
        #outl = Dense(noutclass, activation="softmax")(ll)
        outl = Dense(noutclass, activation="softmax")(gmp1d)
        
        #model = Model(inputs=[tokil], outputs=[outl])
        if sidec:
            model = Model(inputs=[tokil, entil, relentil], outputs=[outl])
        else:
            model = Model(inputs=[tokil], outputs=[outl])
            
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
        self.model = model
        
        bonextil = Input(shape=(None,), dtype='int32')
        boprevil = Input(shape=(None,), dtype='int32')
        bonel = emb_tensor(bonextil)
        boncatted = concatenate([fwd_ll, bonel])
        boncompl = TimeDistributed(Dense(300, activation='relu'))(boncatted)
        bonoutl = TimeDistributed(Dense(1, activation='sigmoid'))(boncompl)
        bopel = emb_tensor(boprevil)
        bopcatted = concatenate([back_ll_r, bopel])
        bopcompl = TimeDistributed(Dense(300, activation='relu'))(bopcatted)
        bopoutl = TimeDistributed(Dense(1, activation='sigmoid'))(bopcompl)
        bonusmodel = Model(inputs=[tokil, bonextil, boprevil], outputs=[bonoutl, bopoutl])
        bonusmodel.compile(loss='binary_crossentropy', optimizer='rmsprop')
        
        train_batches = self.make_batches(self.corpus.train, True)
        test_batches = self.make_batches(self.corpus.test, True)
        eval_batches = self.make_batches(self.corpus.eval, True)
        
        bonusf = open(bonusfilename, "r", encoding="utf-8")
        #bonusf = open("/data/raw/patents/catted_pat_ft_shuffle.txt", "r", encoding="utf-8")
        self.bonusf = bonusf
        
        startn = 0
        for epoch in range(100):
            print("Epoch", epoch, "at", datetime.now())
            if do_transfer:
                if epoch < 5:
                    for subepoch in range(25):
                        bsize = 32
                        bonussents = []
                        # increase to 30000 or higher once we're done
                        # 60000
                        #while(len(bonussents) < 6000):
                        while(len(bonussents) < 12000):
                        #while(len(bonussents) < 50000):
                            l = bonusf.readline()
                            if l == "":
                                print("Wrap around", file=sys.stderr)
                                bonusf = open(bonusfilename, "r", encoding="utf-8")
                                self.bonusf = bonusf
                                # Skip the test head
                                for i in range(nathead): bonusf.readline()
                            ll = l.strip().split()
                            if(len(ll) < 2): continue
                            if(len(ll) > 1000): continue
                            bonussents.append(ll)
                        losses = []
                        catl = []
                        for i in bonussents: catl.extend([tokdict[j] for j in i])
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
                                seq = [tokdict[j] for j in i]
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

            
            n = 0
            print(len(train_batches))
            for b in train_batches:
                #print(b.X.shape, b.X1.shape, b.X2.shape)
                #try:
                if sidec:
                    model.fit(x=[b.X, b.X1, b.X2], y=b.Y, verbose=0)
                else:
                    model.fit(x=[b.X], y=b.Y, verbose=0)
                #except:
                #    pass
                    #print("E", end="")
                    #print(b.X.shape, b.X1.shape, b.X2.shape)
                #model.fit(x=b.X, y=b.Y, verbose=0, class_weight={0:0.035, 1:1.0})
                if n % 50 == 0:
                    print(".", end="")
                    sys.stdout.flush()
                n += 1
                #if n == 100: break
            startn += 100
            print()
            print(datetime.now())
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            tot = 0
            mis = 0
            n = 0
            confmat = [[0 for i in range(6)] for i in range(6)]
            losses = []
            posr = []
            negr = []
            scores = []
            for b in test_batches:
                try:
                    if sidec:
                        pred = model.predict(x=[b.X, b.X1, b.X2])
                    else:
                        pred = model.predict(x=[b.X])
                    for i in range(len(b.Y)):
                        gs = max(range(noutclass), key=lambda x: b.Y[i][x])
                        pp = max(range(noutclass), key=lambda x: pred[i][x])
                        confmat[gs][pp] += 1
                        #print(gs, pp, pred[i])
                        if gs == 0:
                            negr.append(1.0-pred[i][0])
                        else:
                            posr.append(1.0-pred[i][0])
                        losses.append(-math.log(pred[i][gs]))
                        #print(Y[i], pred[i], gs, pp)
                        #scores.append((1.0-pred[i][0], 0 if b.Y[i][0] == 1 else 1))
                        scores.append((1.0-pred[i][0], gs))
                        
                        tot += 1
                        if gs == pp:
                            if gs == 0:
                                tn += 1
                            else:
                                tp += 1
                        elif pp == 0:
                            fn += 1
                        elif gs == 0:
                            fp += 1
                        else:
                            mis += 1
                            fn += 1
                            fp += 1
                except:
                    print("E", end="")
                    raise
                if n % 50 == 0:
                    print(".", end="")
                    sys.stdout.flush()
                n += 1
                #if n == 100: break
            print()
            print(len(posr), len(negr), np.mean(posr), np.mean(negr), np.std(posr), np.std(negr), cohend(posr, negr), ttest_ind(posr, negr))
            print(np.mean(losses))
            print(tp, fp, fn, tn, tot, mis)
            for i in range(6):
                    print("\t".join([str(j) for j in confmat[i]]))
            if tp > 0:
                print(tp/(tp+fp), tp/(tp+fn), 2.0*tp/(tp+tp+fp+fn))
                #print(X1.shape, X2.shape, X3.shape, Y.shape)
                #print(model.evaluate(x=[X1, X2, X3], y=Y))
            sscores = sorted(scores, key=lambda x: -x[0])
            stp = 0.0
            sfp = 0.0
            fmax = 0.0
            bestthresh = 0.0
            sfn = len([i for i in sscores if i[1] > 0])
            precs = []
            for ss in sscores:
                if ss[1] > 0:
                    stp += 1
                    sfn -= 1
                    precs.append(stp/(stp+sfp))
                    f = (2.0*stp/(stp+stp+sfp+sfn))
                    if f > fmax:
                        fmax=f
                        bestthresh = ss[0]
                else:
                    sfp += 1
                #print(ss, stp, sfp, fmax)
            print(np.mean(precs), fmax, bestthresh)
            print("Evaluated at", datetime.now())
            
            if runname is not None:
                n = 0
                outf=open("epoch%s_%s.tsv" % (epoch, runname), "w", encoding="utf-8")
                for b in eval_batches:
                    if sidec:
                        pred = model.predict(x=[b.X, b.X1, b.X2])
                    else:
                        pred = model.predict(x=[b.X])
                    for i in range(len(b.Y)):
                        pp = max(range(noutclass), key=lambda x: pred[i][x])
                        if pp > 0:
                            inst = b.instances[i]
                            print(inst.a.id, corpus.reltypes[pp], "Arg1:%s" % inst.e1, "Arg2:%s" % inst.e2, sep="\t", file=outf)
                    if n % 50 == 0: 
                        print(".", end="")
                        sys.stdout.flush()
                    n += 1
                outf.close()
                print("Submittable at", datetime.now())
