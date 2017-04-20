import urllib.request
import shutil
import sys
import os
from datetime import datetime

def tobits(bit, bits):
	"""Make a one-high encoding.
	
	Args:
		bit: which bit should be high
		bits: how many bits there are
	
	Returns:
		List of 0 and 1 values
	"""
	
	return([1 if i == bit else 0 for i in range(bits)])
		
def sobie_scores_to_char_ents(seq, threshold, ss):
	"""Find all possible entities in a sequence of SOBIE tag score distributions, where the minimum score in the
	entity is greater or equal than the threshold.
	
	Args:
		seq: Dictionary. seq["tokens"] = sequence of token strings. seq["tagfeat"] = 2D array, dim 1=position, dim 2=
		S, O, I, E, B (reverse alphabetical)
		threshold: the lowest permissible value in seq["tagfeat"] for the relevant tag/token combos.
		ss: "sentence string" - the string form of the sentence.
		
	Returns:
		tuple: (
			List of entities - each entity is a tuple, (entity type, start character position, end character position),
			Dictionary of xents - from entity tuple to dictionary of additional values:
				ent - the entity
				pseq - the sequence of scores for the relevant tags in the entity
				oseq - as pseq, but for "O" in each position
				score - the minimum of pseq
				str - the string for the entity
				dom - whether the entity is "dominant" - i.e. not overlapping with a higher-scoring entity
		)
	"""
	
	l = len(seq["tokens"])
	ents = []
	xents = {}
	# Start token
	for i in range(l):
		# End token
		for j in range(i, l):
			# S is special
			if i == j:
				if seq["tagfeat"][i][0] > threshold:
					ent = ("E", seq["tokstart"][i], seq["tokend"][j])
					ents.append(ent)
					xe = {}
					xe["ent"] = ent
					xe["pseq"] = [seq["tagfeat"][i][0]] # Score for S
					xe["oseq"] = [seq["tagfeat"][i][1]] # Score for O
					xe["score"] = xe["pseq"][0] # Score for S
					xe["str"] = ss[ent[1]:ent[2]]
					xents[ent] = xe
			else:
				try:
					# Score for B, then for some number of I, then for E
					pseq = [seq["tagfeat"][i][4]] + [k[2] for k in seq["tagfeat"][i+1:j]] + [seq["tagfeat"][j][3]]
					# Score for some number of O
					oseq = [k[1] for k in seq["tagfeat"][i:j+1]]
				except:
					print(len(seq["tagfeat"]), i, j)
					raise Exception
				if min(pseq) > threshold:
					ent = ("E", seq["tokstart"][i], seq["tokend"][j])
					ents.append(ent)
					xe = {}
					xe["pseq"] = pseq
					xe["oseq"] = oseq
					xe["score"] = min(pseq)
					xe["str"] = ss[ent[1]:ent[2]]
					xents[ent] = xe
				# Check: if the score for I is below threshold, then all longer entities starting at this
				# position will be below threshold, so stop looking.
				# TODO: streamline by also checking B.
				if seq["tagfeat"][j][2] <= threshold: break
	# OK, now we have the entities, mark them dominance. Start with the best scoring entities...
	se = sorted(ents, key=lambda x:-xents[x]["score"])
	# Make a list of which character positions contain dominant entities - none yet
	uu = [False for i in range(len(ss))]
	# For each entity
	for e in se:
		# Dominant unless proved otherwise
		dom = True
		# Are the characters taken?
		for i in range(e[1],e[2]):
			if uu[i]:
				dom = False
				break
		xents[e]["dom"] = dom
		# If dominant, mark those characters as taken
		if dom:
			for i in range(e[1], e[2]): uu[i] = True
	return ents, xents

def bio_to_sobie(seq):
	"""Convert BIO tags to SOBIE tags
	
	Args:
		seq: list of BIO tags

	Returns:
		list of SOBIE tags
	"""
		
	outseq = []
	for i in range(len(seq)):
		t = seq[i]
		prev = "O" if i == 0 else seq[i-1]
		next = "O" if i == len(seq)-1 else seq[i+1]
		typ = t[1:]
		ntyp = next[1:]
		ptyp = prev[1:]
		
		if t.startswith("B"):
			if next == "I%s" % typ:
				outseq.append(t)
			else:
				outseq.append("S%s" % typ)
		elif t.startswith("I"):
			if next == "O" or ntyp != typ or next.startswith("B"):
				outseq.append("E%s" % typ)
			else:
				outseq.append("I%s" % typ)
		else:
			outseq.append("O")
	return outseq

def get_file(filename):
	dir = os.path.expanduser(os.path.join("~", ".chemlistem"))
	if not os.path.exists(dir): os.makedirs(dir)
	f = os.path.join(dir, filename)
	if os.path.exists(f): return f
	origin = "https://bitbucket.org/rscapplications/chemlistem/downloads/"
	url = origin + filename
	tmpf = os.path.join(dir, "tmp")
	print("Fetching", url, "at", datetime.now(), file=sys.stderr)
	with urllib.request.urlopen(url) as resp, open(tmpf, "wb") as outf:
		shutil.copyfileobj(resp, outf)
	os.rename(tmpf, f)
	print("Fetched", url, "at", datetime.now(), file=sys.stderr)
	return f