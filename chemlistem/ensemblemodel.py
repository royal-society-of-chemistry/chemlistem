import sys
from collections import defaultdict
from datetime import datetime
from .tradmodel import TradModel, get_trad_model
from .minimodel import MiniModel, get_mini_model

defaultmodel = None
defaultmodelgpu = False

def get_ensemble_model(version="0.1.0", gpu=False):
    """
    Gets the default ensemble model - by getting the constituent models, downloading if necessary.
    
    Args:
        version: the version number on BitBucket.
        gpu: whether to use CuDNNLSTM.
    
    Returns:
        An EnsembleModel
    """
    global defaultmodel, defaultmodelgpu
    if defaultmodel is not None and gpu == defaultmodelgpu: return defaultmodel
    defaultmodelgpu = gpu
    em = EnsembleModel(get_trad_model(version, gpu), get_mini_model(version, gpu))
    defaultmodel = em
    return defaultmodel

class EnsembleModel(object):
    """
    A simple ensemble, consisting of a TradModel and a MiniModel.
    """

    def __init__(self, tradmodel, minimodel):
        """
        Set up the ensemble.
        
        Args:
            tradmodel: the TradModel
            minimodel: the MiniModel
        """
        self.tradmodel = tradmodel
        self.minimodel = minimodel
        print("Ensemble Model ready at", datetime.now(), file=sys.stderr)
        
    def process(self, str, threshold=0.475, domonly=True):
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
        subthresh = threshold / 10
        r1 = self.tradmodel.process(str, subthresh, False)
        r2 = self.minimodel.process(str, subthresh, False)
        pos_to_ents = defaultdict(list)
        for e in r1:
            pos = (e[0], e[1])
            pos_to_ents[pos].append(e)
        for e in r2:
            pos = (e[0], e[1])
            pos_to_ents[pos].append(e)
        nents = []
        for e in pos_to_ents:
            score = sum([i[3] for i in pos_to_ents[e]]) / 2
            if score >= threshold:
                nents.append([e[0], e[1], pos_to_ents[e][0][2], score, False])
        se = sorted(nents, key=lambda x:-x[3])
        uu = [False for i in range(len(str))]
        for e in se:
            dom = True
            for i in range(e[0],e[1]):
                if uu[i]:
                    dom = False
                    break
            if dom:
                for i in range(e[0], e[1]): uu[i] = True
                e[4] = True
        if domonly: nents = [i for i in nents if i[4]]
        nents = sorted([tuple(i) for i in nents])
        return nents
    
    def batchprocess(self, instrs, threshold=0.475, domonly=True):
        subthresh = threshold / 10
        rr1 = self.tradmodel.batchprocess(instrs, subthresh, False)
        rr2 = self.minimodel.batchprocess(instrs, subthresh, False)
        res = []
        for n in range(len(rr1)):
            r1 = rr1[n]
            r2 = rr2[n]
            pos_to_ents = defaultdict(list)
            for e in r1:
                pos = (e[0], e[1])
                pos_to_ents[pos].append(e)
            for e in r2:
                pos = (e[0], e[1])
                pos_to_ents[pos].append(e)
            nents = []
            for e in pos_to_ents:
                score = sum([i[3] for i in pos_to_ents[e]]) / 2
                if score >= threshold:
                    nents.append([e[0], e[1], pos_to_ents[e][0][2], score, False])
            se = sorted(nents, key=lambda x:-x[3])
            uu = [False for i in range(len(instrs[n]))]
            for e in se:
                dom = True
                for i in range(e[0],e[1]):
                    if uu[i]:
                        dom = False
                        break
                if dom:
                    for i in range(e[0], e[1]): uu[i] = True
                    e[4] = True
            if domonly: nents = [i for i in nents if i[4]]    
            nents = sorted([tuple(i) for i in nents])
            res.append(nents)
        return res
        