import numpy as np
import string
import gzip
import os
import sys
from pathlib import Path

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
    is_ignore = False
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
    print ("USAGE: python make_paired_MSA_simple.py [a3m*]")
    sys.exit()

tags = []
query = {}
a3m = {}
for i,fn in enumerate(sys.argv[1:]):
    tag = Path(fn).stem+'_'+str(i)
    tags.append(tag)
    #print ('Read',fn,'into',tag)
    query[tag],a3m[tag] = read_a3m(fn)

#wrt = '> query\n'
#wrt += '/'.join([query[i] for i in tags])+'\n'
paired_data = []
paired_data.append( (9999,'query','/'.join([query[i] for i in tags])) )


marked = {}
for i in range(len(tags)):
    fn1 = tags[i]

    preseq = ''
    for pre in range(i):
        if (pre>0):
            preseq += '/'
        preseq += '-'*len(query[ tags[pre] ])

    for tax in a3m[fn1]:
        name = a3m[fn1][tax][0]
        if (i>0):
            seq = preseq+'/'+a3m[fn1][tax][1]
        else:
            seq = a3m[fn1][tax][1]
        ct = 1

        if (fn1+'.'+tax in marked):
            continue

        for j in range(i+1,len(tags)):
            fn2 = tags[j]
            if tax in a3m[fn2]:
                name += ' '+a3m[fn2][tax][0]
                seq += '/'
                seq += a3m[fn2][tax][1]
                marked[fn2+'.'+tax] = 1
                ct+=1
            else:
                seq += '/'
                seq += '-'*len(query[ fn2 ])

        marked[fn1+'.'+tax] = 1
        paired_data.append( (ct,name,seq) )

        #wrt += '>'+name+'\n'
        #wrt += seq+'\n'

paired_data = sorted(paired_data, key=lambda x: x[0], reverse=True)
for p in paired_data:
    print ('>',p[1])
    print (p[2])
