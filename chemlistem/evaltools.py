from .corpusreader import CorpusReader

from collections import Counter

def internal_eval(textfile, annotfile, model):
    cmplogf = open("cmplogc.txt", "w", encoding="utf-8")
    cr = CorpusReader(textfile, annotfile)
    tp = 0
    fp = 0
    fn = 0
    for n, s in enumerate(cr.testseqs):
        ss = s["ss"]
        ents = s["ents"]
        entset = {(i[2],i[3]) for i in ents}
        procents = model.process(ss, logf=cmplogf)
        for i in procents:
            print(n, i, file=cmplogf)
        me = [i for i in procents if (i[0], i[1]) in entset]
        #xme = [i for i in procents if i not in me]
        #print(entset,procents,me,xme)
        xtp = len(me)
        xfp = len(procents)-xtp
        xfn = len(entset)-xtp
        #print(xtp, xfp, xfn)
        tp += xtp
        fp += xfp
        fn += xfn
    cmplogf.close()
    return (tp,fp,fn,tp*1.0/(tp+fp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+fp+fn))

def internal_eval_batch(textfile, annotfile, model):
    cmplogf = open("cmplogc.txt", "w", encoding="utf-8")
    cr = CorpusReader(textfile, annotfile)
    tp = 0
    fp = 0
    fn = 0
    sentences = [i["ss"] for i in cr.testseqs]
    procentss = model.batchprocess(sentences)
    for n, s in enumerate(cr.testseqs):
        ents = s["ents"]
        entset = {(i[2],i[3]) for i in ents}
        procents = procentss[n]
        for i in procents:
            print(n, i, file=cmplogf)
        me = [i for i in procents if (i[0], i[1]) in entset]
        #xme = [i for i in procents if i not in me]
        #print(entset,procents,me,xme)
        xtp = len(me)
        xfp = len(procents)-xtp
        xfn = len(entset)-xtp
        #print(xtp, xfp, xfn)
        tp += xtp
        fp += xfp
        fn += xfn
    cmplogf.close()
    return (tp,fp,fn,tp*1.0/(tp+fp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+fp+fn))
                      
def external_eval(textfile, model,outfn):
    outf = open(outfn, "w", encoding="utf-8", errors="replace")
    print("DOCUMENT_ID\tSECTION\tINIT\tEND\tSCORE\tANNOTATED_TEXT\tTYPE\tDATABASE_ID", file=outf)
    inf = open(textfile, encoding="utf-8", errors="replace")
    pairs = []
    for l in inf:
        ll = l.strip().split("\t")
        pairs.append((ll[0],"T",ll[1]))
        pairs.append((ll[0],"A",ll[2]))
    en = 1
    for p in pairs:
        pe = model.process(p[2])
        for e in pe:
            print("\t".join((p[0],p[1],"%s"%e[0],"%s"%e[1],"%s"%e[3],e[2],"unknown","%s"%en)), file=outf)
            en += 1
    outf.close()
    
def external_eval_batch(textfile, model,outfn):
    outf = open(outfn, "w", encoding="utf-8", errors="replace")
    print("DOCUMENT_ID\tSECTION\tINIT\tEND\tSCORE\tANNOTATED_TEXT\tTYPE\tDATABASE_ID", file=outf)
    inf = open(textfile, encoding="utf-8", errors="replace")
    pairs = []
    for l in inf:
        ll = l.strip().split("\t")
        pairs.append([ll[0],"T",ll[1]])
        pairs.append([ll[0],"A",ll[2]])
    items = [i[2] for i in pairs]
    procd = model.batchprocess(items)
    for p, i in zip(pairs, procd):
        p.append(i)
    en = 1
    for p in pairs:
        #print(p)
        pe = p[3]
        for e in pe:
            print("\t".join((p[0],p[1],"%s"%e[0],"%s"%e[1],"%s"%e[3],e[2],"unknown","%s"%en)), file=outf)
            en += 1
    outf.close()
    
def cmp_eval_files(efile, gsfile):
    eents = set()
    gsents = set()
    for ents, fn in [(eents, efile), (gsents, gsfile)]:
        f = open(fn, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split("\t")
            if ll[0] == "DOCUMENT_ID": continue # skip headers
            ents.add((ll[0], ll[1], ll[2], ll[3]))
    jents = eents & gsents
    tp = len(jents)
    fp = len(eents) - tp
    fn = len(gsents) - tp
    return (tp, fp, fn, tp*1.0/(tp+fp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+fp+fn))

def cmp_eval_files_multi(efile, gsfile):
    eents = set()
    gsents = set()
    for ents, fn in [(eents, efile), (gsents, gsfile)]:
        f = open(fn, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split("\t")
            if ll[0] == "DOCUMENT_ID": continue # skip headers
            ents.add((ll[0], ll[1], ll[2], ll[3]))
            
    
    jents = eents & gsents
    tp = len(jents)
    fp = len(eents) - tp
    fn = len(gsents) - tp
    r1 = (tp, fp, fn, tp*1.0/(tp+fp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+fp+fn))
    goodpapers = {i[0] for i in gsents}
    good_eents = {i for i in eents if i[0] in goodpapers}
    nfp = len(good_eents) - tp
    r2 = (tp, nfp, fn, tp*1.0/(tp+nfp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+nfp+fn))
    return (r1, r2)

def cmp_eval_files_x(efile, gsfile):
    eents = set()
    gsents = set()
    sd = {}
    epc = Counter()
    gspc = Counter()
    
    for ents, fn, pc in [(eents, efile, epc), (gsents, gsfile, gspc)]:
        f = open(fn, "r", encoding="utf-8")
        for l in f:
            ll = l.strip().split("\t")
            if ll[0] == "DOCUMENT_ID": continue # skip headers
            ents.add((ll[0], ll[1], ll[2], ll[3]))
            if fn == efile: sd[(ll[0], ll[1], ll[2], ll[3])] = float(ll[4])
            pc[ll[0]] += 1
    jents = eents & gsents
    tp = len(jents)
    fp = len(eents) - tp
    fn = len(gsents) - tp
    xents = eents - jents
    xes = sorted(xents, key=lambda x:-sd[x])
    #for i in xes:
    #    print(i, sd[i])
    for i in [j for j in epc if j not in gspc]:
        print(i, epc[i])
    print(sum([epc[i] for i in epc if i not in gspc]))
    
    
    return (tp, fp, fn, tp*1.0/(tp+fp),tp*1.0/(tp+fn),tp*2.0/(tp+tp+fp+fn))
    
                            