import re
import io
import sys
import time
import json

defre = re.compile("\"(.*)\" \\[(.*)\\]")
xrefre = re.compile("([^\"]+?)( \"(.*)\")?")
relre = re.compile("([a-z_]+) (CHEBI:\d+)( ! (.*))?")
synre = re.compile("\"(.*)\" ([A-Za-z_ ]+) \\[(.*?):?\\]")

def _unescape(s):
	return (s.replace("\\\"", "\"")
		.replace("\\W", " ")
		.replace("\\t", "\t")
		.replace("\\:", ":")
		.replace("\\,", ",")
		.replace("\\(", "(")
		.replace("\\)", ")")
		.replace("\\[", "[")
		.replace("\\]", "]")
		.replace("\\{", "{")
		.replace("\\}", "}")
		.replace("\n", "")
		.replace("\\\n", "\n")
		.replace("\\\\", "\\"))

class ChebiOBO(object):
	"""
	A representation of the ChEBI ontology.
	
	Used to generate chebinames.txt for chemlistem.
	"""
	def __init__(self, fname):
		f = open(fname, "r", encoding="utf-8")
		
		tt = time.time()
		terms = []
		self.terms = terms
		term = None
		for line in f:
			line = line.strip()
			if line == "[Term]":
				term = {}
				term["id"] = ""
				term["name"] = ""
				term["def"] = ""
				term["subset"] = ""
				term["synonym"] = []
				term["xref"] = []
				term["is_a"] = []
				term["relationship"] = []
				term["alt_id"] = []
				term["is_obsolete"] = False
				terms.append(term)
			elif line == "":
				term = None
			elif term == None:
				pass
			elif line == "[Typedef]":
				term = None
			elif not ":" in line:
				print("Argh!", line)
			else:
				t, content = line.split(": ", 1)
				if t not in term: print(line)
				if t == "id":
					term["id"] = content
				elif t == "name":
					term["name"] = _unescape(content)
				elif t == "is_obsolete":
					term["is_obsolete"] = content == "true"
				elif t == "def":
					m = defre.match(content)
					term["def"] = m.group(1)
				elif t == "alt_id":
					term["alt_id"].append(content)
				elif t == "is_a":
					term["is_a"].append(content.split(" ! ")[0])
				elif t == "xref":
					m = xrefre.match(content)
					if m is None:
						print(t, content)
					if m.group(3) is None:
						xr = {"xref": m.group(1), "type": ""}
					else:
						xr = {"xref": m.group(1), "type": _unescape(m.group(3))}
					term["xref"].append(xr)
				elif t == "relationship":
					m = relre.match(content)
					if m is None:
						print(t, content)
					if m.group(4) is None:
						rel = {"rel": m.group(1), "id": m.group(2), "comment": ""}					
					else:
						rel = {"rel": m.group(1), "id": m.group(2), "comment": m.group(4)}
					term["relationship"].append(rel)
				elif t == "synonym":
					m = synre.match(content)
					if m is None:
						print(t, content)
					syn = {"syn": _unescape(m.group(1)), "type": m.group(2), "supplier": m.group(3)}
					term["synonym"].append(syn)
				elif t == "subset":
					term["subset"] = content
			
		print("Loaded in", time.time() - tt)
		print(len(terms))
		ididx = {}
		self.ididx = ididx
		tt = time.time()
		nameidx = {}
		self.nameidx = nameidx
		for term in terms:
			ididx[term["id"]] = term
			term["rev_is_a"] = []
			term["rev_relationship"] = []
		for term in terms:
			for altid in term["alt_id"]:
				if altid in ididx: print(altid)
				ididx[altid] = term
			for id in term["is_a"]:
				ididx[id]["rev_is_a"].append(term["id"])
			for rel in term["relationship"]:
				nrel = rel.copy()
				nrel["id"] = term["id"]
				ididx[rel["id"]]["rev_relationship"].append(nrel)
			for syn in term["synonym"]:
				term["goodsyns"] = []
				if syn["type"] == "RELATED InChI":
					term["InChI"] = syn["syn"]
				elif syn["type"] == "RELATED SMILES":
					term["SMILES"] = syn["syn"]
				elif syn["type"] == "RELATED InChIKey":
					term["InChIKey"] = syn["syn"]
				elif syn["type"] == "FORMULA":
					term["formula"] = syn["syn"]
				else:
					term["goodsyns"].append(syn["syn"])
					nameidx[syn["syn"]] = term
		for term in terms:
			nameidx[term["name"]] = term
		print("Indexed in", time.time()-tt)
		f.close()
			
def make_chebinames_file(infn, outfn):
	"""Make a list of chemical names from ChEBI.
	
	Args:
		infn: input filename e.g. "chebi.obo"
		outfn: output filename e.g. "chebinames.txt"
	"""
	co = ChebiOBO(infn)
	f = open(outfn, "w", encoding="utf-8")
	for term in co.terms:
		if "InChI" not in term: continue
		f.write(term["name"]+"\n")
		if "goodsyns" in term:
			for s in term["goodsyns"]:
				f.write(s+"\n")
	f.close()
