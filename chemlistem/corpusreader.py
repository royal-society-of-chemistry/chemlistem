import time
import io
import sys
import os
import json
import random
from datetime import datetime
from chemtok.ChemTokeniser import ChemTokeniser
from .utils import bio_to_sobie
	
class CorpusReader(object):
	"""
	Reads, tokenises and finds entities in BioCreative V.5 CEMP training data. Splits the
	data into two parts - 4/5 training, 1/5 test.
	
	Members:
	trainseqs - train sequences
	testseqs - test sequences
	
	Each seq object is a dictionary, containing:
	"tokens": the tokens, as a list of strings
	"tokstart": the start offsets for each token, as a list
	"tokend": the end offsets for each token, as a list
	"bio": BIO or SOBIE tags
	"ss": the string for the sequence 
	"ents": a list of tuples, one per entity, corresponding to the six fields in the annotations file eg:
		CA2073855C	A	834	843	Alkaloids	FAMILY
		PatentID, TitleOrAbstract, StartOffset, EndOffset, String, Type
		
	"""
	
	def __init__(self, textfile, annotfile, tosobie=True, charbychar=False):
		"""
		Args:
			textfile: the filename of the file containing the text - e.g. "BioCreative V.5 training set.txt"
			annotfile: the filename of the file containing the annotations - e.g. "CEMP_BioCreative V.5 training set annot.tsv"
		
		"""
		print("Reading corpus at", datetime.now())
		self.aggressive = False
		self.charbychar = charbychar
		self.alle = True # convert all entity types to "E"
		self.tosobie = tosobie
	
		trainlines, testlines = self.shufflesplitlines(textfile)
	
		print("Read annots at", datetime.now())
		f = open(annotfile, "r", encoding="utf-8", errors="replace")
		self.items = []
		self.items_by_text = {}
		for l in f:
			ll = l.strip().split("\t")
			item = (ll[0], ll[1], int(ll[2]), int(ll[3]), ll[4], ll[5])
			self.items.append(item)
			itext = (ll[0], ll[1])
			if itext not in self.items_by_text: self.items_by_text[itext] = []
			self.items_by_text[itext].append(item)
		f.close()
		
		print("Read train seqs at", datetime.now())
		self.split_on_boundaries = True
		self.trainseqs = []
		for l in trainlines:
			ll = l.strip().split("\t")
			t1 = (ll[0], "T")
			t2 = (ll[0], "A")
			self.trainseqs.append(self._toBIO(ll[1], t1, True))
			self.trainseqs.append(self._toBIO(ll[2], t2, True))
		print("Read test seqs at", datetime.now())
		self.split_on_boundaries = False
		self.testseqs = []
		for l in testlines:
			ll = l.strip().split("\t")
			t1 = (ll[0], "T")
			t2 = (ll[0], "A")
			self.testseqs.append(self._toBIO(ll[1], t1, False))
			self.testseqs.append(self._toBIO(ll[2], t2, False))
		print("Corpus read at", datetime.now())


	def _toBIO(self, text, textid, split_on_boundaries):
		ct = ChemTokeniser(text, aggressive=self.aggressive, charbychar=self.charbychar, clm=True)
		
		if split_on_boundaries and textid in self.items_by_text:
			boundaries = []
			for i in self.items_by_text[textid]:
				boundaries.append(i[2])
				boundaries.append(i[3])
			boundaries = sorted(boundaries)
			bptr = 0
			tokptr = 0
			while tokptr < len(ct.tokens) and bptr < len(boundaries):
				tok = ct.tokens[tokptr]
				if tok.end < boundaries[bptr]:
					tokptr += 1
				elif tok.end == boundaries[bptr]:
					tokptr += 1
					bptr += 1
				elif tok.start >= boundaries[bptr]:
					bptr += 1
				else:
					tsplit = tok.splitAt(boundaries[bptr])
					ct.tokens = ct.tokens[:tokptr] + tsplit + ct.tokens[tokptr+1:]
			ct.numberTokens()	
				
		toksbystart = {tok.start:tok for tok in ct.tokens}
		toksbyend = {tok.end:tok for tok in ct.tokens}
		labels = ["O" for tok in ct.tokens]
		starts = [tok.start for tok in ct.tokens]
		ends = [tok.end for tok in ct.tokens]
		seq = {"tokens": [i.value for i in ct.tokens], "tokstart": starts, "tokend": ends, "bio": labels}
		if textid in self.items_by_text:
			items = self.items_by_text[textid]
			etype = "E" if self.alle else i[5]
			for i in items:
				if i[2] in toksbystart and i[3] in toksbyend:
					startid = toksbystart[i[2]].id
					endid = toksbyend[i[3]].id
					labels[startid] = "B-" + etype
					if endid > startid:
						for j in range(startid+1, endid+1):
							labels[j] = "I-" + etype
		else:
			items = []
		seq["ents"] = items
		if self.tosobie:
			seq["bio"] = bio_to_sobie(seq["bio"])
		seq["ss"] = text
		return seq
		
		
	def shufflesplitlines(self, textfile):
		print("Shufflesplit at", datetime.now())
		f = open(textfile, "r", encoding="utf-8", errors="replace")
		lines = []
		for l in f:
			lines.append(l.strip())
		random.seed(0)
		random.shuffle(lines)
		splitpoint = int(len(lines)*4/5)
		return lines[:splitpoint], lines[splitpoint:]
		
if __name__=="__main__":
	print(datetime.now())
	cr = CorpusReader("BioCreative V.5 training set.txt", "CEMP_BioCreative V.5 training set annot.tsv")
	#cr = CorpusReader("chemdner_patents_train_text.txt", "chemdner_cemp_gold_standard_train.tsv")
	print(datetime.now())
	print(len(cr.testseqs), len(cr.trainseqs))
