import numpy as np
import string
import gzip
import os
import sys

TABLE = str.maketrans(dict.fromkeys(string.ascii_lowercase))
ALPHABET = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)

def seq2number(seq):
    seq_no_ins = seq.translate(TABLE)
    seq_no_ins = np.array(list(seq_no_ins), dtype='|S1').view(np.uint8)
    for i in range(ALPHABET.shape[0]):
        seq_no_ins[seq_no_ins == ALPHABET[i]] = i
    seq_no_ins[seq_no_ins > 20] = 20

    return seq_no_ins

def calc_seqID(query, cand):
    same = (query == cand).sum()
    return same / float(len(query))

def read_a3m(fn):
    # read sequences in a3m file
    # only take one (having the highest seqID to query) per each taxID
    is_first = True
    tmp = {}
    if fn.split('.')[-1] == "gz":
        fp = gzip.open(fn, 'rt')
    else:
        fp = open(fn, 'r')

    for line in fp:
        if line[0] == ">":
            if is_first:
                continue
            x = line.split()
            seqID = x[0][1:]
            try:
                idx = line.index("TaxID")
                is_ignore = False
            except:
                is_ignore = True
                continue
            TaxID = line[idx:].split()[0].split('=')[-1]
            if not TaxID in tmp:
                tmp[TaxID] = list()
        else:
            if is_first:
                query = line.strip()
                is_first = False
            elif is_ignore:
                continue
            else:
                tmp[TaxID].append((seqID, line.strip()))

    query_in_num = seq2number(query)
    a3m = {}
    for TaxID in tmp:
        if len(tmp[TaxID]) < 1:
            continue
        if len(tmp[TaxID]) < 2:
            a3m[TaxID] = tmp[TaxID][0]
            continue
        # Get the best sequence only
        score_s = list()
        for seqID, seq in tmp[TaxID]:
            seq_in_num = seq2number(seq)
            score = calc_seqID(query_in_num, seq_in_num)
            score_s.append(score)
        #
        idx = np.argmax(score_s)
        a3m[TaxID] = tmp[TaxID][idx]

    return query, a3m

if len(sys.argv) == 1:
    print ("USAGE: python make_paired_MSA_simple.py [a3m for chain A] [a3m for chain B] [output filename]")
    sys.exit()

fnA = sys.argv[1]
fnB = sys.argv[2]
pair_fn = sys.argv[3]
#
queryA, a3mA = read_a3m(fnA)
queryB, a3mB = read_a3m(fnB)
wrt = '>query\n'
wrt += queryA
wrt += '/'
wrt += queryB
wrt += "\n"
for taxA in a3mA:
    if taxA in a3mB:
        wrt += ">%s %s\n"%(a3mA[taxA][0], a3mB[taxA][0])
        wrt += "%s/%s\n"%(a3mA[taxA][1], a3mB[taxA][1])

with open(pair_fn, 'wt') as fp:
    fp.write(wrt)
