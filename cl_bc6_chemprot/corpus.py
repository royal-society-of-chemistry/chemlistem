import chemtok
import numpy as np
import random
from collections import defaultdict

reltypes = ["NONE", "CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"]
reltyped = {k: v for v, k in enumerate(reltypes)}

fullreltypes = ["NONE", "CPR:1", "CPR:2", "CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:7", "CPR:8", "CPR:9", "CPR:10"]
fullreltyped = {k: v for v, k in enumerate(fullreltypes)}

fullrelsubtypes = ["NONE", "PART-OF", "REGULATOR", "DIRECT-REGULATOR", "INDIRECT-REGULATOR", "UPREGULATOR",
                   "ACTIVATOR", "INDIRECT-UPREGULATOR", "DOWNREGULATOR", "INHIBITOR", "INDIRECT-DOWNREGULATOR",
                   "AGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "ANTAGONIST", "MODULATOR", "MODULATOR-ACTIVATOR",
                   "MODULATOR-INHIBITOR", "COFACTOR", "SUBSTRATE", "PRODUCT-OF", "SUBSTRATE-PRODUCT-OF", "NOT"]
fullrelsubtyped = {k: v for v, k in enumerate(fullrelsubtypes)}
stm = {"NONE": "NONE",
       "PART-OF": "NONE", 
       "REGULATOR": "NONE", 
       "DIRECT-REGULATOR": "NONE",
       "INDIRECT-REGULATOR": "NONE",
       "UPREGULATOR": "CPR:3",
       "ACTIVATOR": "CPR:3", 
       "INDIRECT-UPREGULATOR": "CPR:3",
       "DOWNREGULATOR": "CPR:4",
       "INHIBITOR": "CPR:4",
       "INDIRECT-DOWNREGULATOR": "CPR:4",
       "AGONIST": "CPR:5", 
       "AGONIST-ACTIVATOR": "CPR:5", 
       "AGONIST-INHIBITOR": "CPR:5",
       "ANTAGONIST": "CPR:6",
       "MODULATOR": "NONE",
       "MODULATOR-ACTIVATOR": "NONE",
       "MODULATOR-INHIBITOR": "NONE",
       "COFACTOR": "NONE",
       "SUBSTRATE": "CPR:9",
       "PRODUCT-OF": "CPR:9",
       "SUBSTRATE-PRODUCT-OF": "CPR:9",
       "NOT": "NONE"}
subtype_to_reltypeno = {i:reltyped[stm[i]] for i in stm}
subtypeno_to_reltypeno = {fullrelsubtyped[i]:reltyped[stm[i]] for i in stm}

typeno_to_reltypeno = defaultdict(lambda: 0)
for i in range(len(fullreltypes)):
    if fullreltypes[i] in reltyped:
        typeno_to_reltypeno[i] = reltyped[fullreltypes[i]]
print(len(reltypes))
print(len(fullreltypes))
print(len(fullrelsubtypes))

class Abstr(object):
    def __init__(self, id, title, abstr):
        self.id = id
        self.title = title
        self.abstr = abstr
        
        self.combined = "%s\t%s" % (title, abstr)
        
        self.tokr = chemtok.ChemTokeniser(self.combined)
        
        
        self.entities = []
        self.edict = {}
        self.gsrelns = []
        self.chems = []
        self.genes = []
        self.posrelnset = set()
        
        self.fullrelns = []
        self.posfullrelnset = set()
                
    def addentity(self, e):
        self.entities.append(e)
        self.edict[e.eid] = e
        if e.etype == "CHEMICAL":
            self.chems.append(e.eid)
        else:
            self.genes.append(e.eid)
        firsttoken = None
        lasttoken = None
        for i in range(len(self.tokr.tokens)):
            tok = self.tokr.tokens[i]
            if firsttoken is None and tok.end > e.startc:
                firsttoken = i
                startpartial = 1 if tok.start != e.startc else 0
            if lasttoken is None and firsttoken is not None and tok.end >= e.endc:
                lasttoken = i
                endpartial = 1 if tok.end != e.endc else 0
                break
        if lasttoken is None or firsttoken is None: print("Argh!")
        e.firsttoken = firsttoken
        e.lasttoken = lasttoken
        e.startpartial = startpartial
        e.endpartial = endpartial
        
        #print(e.value, "*%s*" % self.combined[e.startc:e.endc], "|".join([t.value for t in self.tokr.tokens[firsttoken:lasttoken+1]]), firstpartial, lastpartial)
        
    def addgsreln(self, r):
        self.gsrelns.append(r)
        self.posrelnset.add(r.e1ce2)

    def addfullreln(self, r):
        self.fullrelns.append(r)
        self.posfullrelnset.add(r.e1ce2)    
        
    def addnegrelns(self):
        #print(self.id, self.chems, self.genes)
        for chem in self.chems:
            #print(chem)
            for gene in self.genes:
                e1ce2 = "%s:%s" % (chem, gene)
                #print(e1ce2)
                if e1ce2 not in self.posrelnset:
                    self.addgsreln(GSReln(self.id, "NONE", chem, gene, self))
                if e1ce2 not in self.posfullrelnset:
                    self.addfullreln(FullReln(self.id, "NONE", "NONE", chem, gene, self))
    
    def makearrays(self):
        self.toktyp = np.zeros((len(self.tokr.tokens), 16))
        for i in range(len(self.tokr.tokens)): self.toktyp[i][0] = 1
        typmap = {"CHEMICAL":1, "GENE-Y":6, "GENE-N":11}
        for e in self.entities:
            typeadj = typmap[e.etype]
            for tn in range(e.firsttoken, e.lasttoken+1):
                self.toktyp[tn, typeadj] = 1
                self.toktyp[tn, 0] = 0
            self.toktyp[e.firsttoken, typeadj+1] = 1
            if e.startpartial: self.toktyp[e.firsttoken, typeadj+2] = 1
            self.toktyp[e.lasttoken, typeadj+3] = 1
            if e.endpartial: self.toktyp[e.firsttoken, typeadj+4] = 1
        self.relnx = np.zeros((len(self.gsrelns), len(self.tokr.tokens), 2))
        self.relny = np.zeros((len(self.gsrelns), len(reltypes)))
        
        for i, r in enumerate(self.gsrelns):
            self.relny[i][r.rtypen] = 1
            e1 = r.e1e
            for tn in range(e1.firsttoken, e1.lasttoken+1):
                self.relnx[i][tn][0] = 1
            e2 = r.e2e
            for tn in range(e2.firsttoken, e2.lasttoken+1):
                self.relnx[i][tn][1] = 1
        self.relnx_full = np.zeros((len(self.fullrelns), len(self.tokr.tokens), 2))
        self.relny_full = np.zeros((len(self.fullrelns), len(fullreltypes)))
        self.relny_full_sub = np.zeros((len(self.fullrelns), len(fullrelsubtypes)))
        for i, r in enumerate(self.fullrelns):
            self.relny_full[i][r.rtypen] = 1
            self.relny_full_sub[i][r.rsubtypen] = 1
            e1 = r.e1e
            for tn in range(e1.firsttoken, e1.lasttoken+1):
                self.relnx_full[i][tn][0] = 1
            e2 = r.e2e
            for tn in range(e2.firsttoken, e2.lasttoken+1):
                self.relnx_full[i][tn][1] = 1                


                
        
class Entity(object):
    def __init__(self, aid, eid, etype, startc, endc, value):
        self.aid = aid
        self.eid = eid
        self.etype = etype
        self.startc = startc
        self.endc = endc
        self.value = value
        
    def __repr__(self):
        return "[Entity: %s %s %s]" % (self.value, self.startc, self.endc)
        
class GSReln(object):
    def __init__(self, aid, rtype, e1, e2, abstr):
        self.aid = aid
        self.rtype = rtype
        self.rtypen = reltyped[rtype]
        self.e1 = e1
        self.e2 = e2
        self.e1ce2 = "%s:%s" % (e1, e2)
        self.abstr = abstr
        self.e1e = abstr.edict[e1]
        self.e2e = abstr.edict[e2]
        self.e2yn = 1 if self.e2e.etype.endswith("Y") else 0
    
    def __repr__(self):
        return "[Reln: %s %s %s %s]" % (self.aid, self.rtype, self.e1, self.e2)

class FullReln(object):
    def __init__(self, aid, rtype, rsubtype, e1, e2, abstr):
        self.aid = aid
        self.rtype = rtype
        self.rtypen = fullreltyped[rtype]
        self.rsubtype = rsubtype
        self.rsubtypen = fullrelsubtyped[rsubtype]
        self.evaltype = subtype_to_reltypeno[rsubtype]
        self.e1 = e1
        self.e2 = e2
        self.e1ce2 = "%s:%s" % (e1, e2)
        self.abstr = abstr
        self.e1e = abstr.edict[e1]
        self.e2e = abstr.edict[e2]
        self.e2yn = 1 if self.e2e.etype.endswith("Y") else 0
        
    def __repr__(self):
        return "[FullReln: %s %s %s %s %s]" % (self.aid, self.rtype, self.rsubtype, self.e1, self.e2)
        
class Corpus(object):
    
    def __init__(self):
        self.alist = []
        self.adict = {}
        
        self.train = self.load("training")
        self.test = self.load("development")
        self.eval = self.load("test", False)
        
    def load(self, name, loadrelns=True):
        co = []
        
        f = open("chemprot_%s/chemprot_%s_abstracts.tsv" % (name, name), "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split("\t")
            id, title, abstr = ll
            a = Abstr(id, title, abstr)
            self.alist.append(a)
            self.adict[a.id] = a
            co.append(a)
        
        f = open("chemprot_%s/chemprot_%s_entities.tsv" % (name, name), "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split("\t")
            if not ll[0] in self.adict:
                print(ll[0])
            else:
                a = self.adict[ll[0]]
                e = Entity(ll[0], ll[1], ll[2], int(ll[3]), int(ll[4]), ll[5])
                a.addentity(e)
        
        if loadrelns:
            f = open("chemprot_%s/chemprot_%s_gold_standard.tsv" % (name, name), "r", encoding="utf-8")
            for l in f:
                ll = l.strip().split("\t")
                if not ll[0] in self.adict:
                    print(ll[0])
                else:
                    a = self.adict[ll[0]]
                    r = GSReln(ll[0], ll[1], ll[2].split(":")[1], ll[3].split(":")[1], a)
                    a.addgsreln(r)
            f = open("chemprot_%s/chemprot_%s_relations.tsv" % (name, name), "r", encoding="utf-8")
            for l in f:
                ll = l.strip().split("\t")
                if not ll[0] in self.adict:
                    print(ll[0])
                elif ll[1] == "CPR:0":
                    print(ll)
                else:
                    a = self.adict[ll[0]]
                    r = FullReln(ll[0], ll[1], ll[3].replace("_", "-"), ll[4].split(":")[1], ll[5].split(":")[1], a)
                    a.addfullreln(r)
        
        for a in co:
            a.addnegrelns()
            a.makearrays()
        
        return co