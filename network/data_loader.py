import torch
from torch.utils import data
import os
import csv
from dateutil import parser
import numpy as np
from parsers import parse_a3m, parse_pdb
from chemical import INIT_CRDS
from util import center_and_realign_missing, random_rot_trans
from symmetry import get_symmetry

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
compl_dir = "/projects/ml/RoseTTAComplex"
fb_dir = "/projects/ml/TrRosetta/fb_af"
if not os.path.exists(base_dir):
    # training on blue
    base_dir = "/gscratch2/PDB-2021AUG02"
    compl_dir = "/gscratch2/RoseTTAComplex"
    fb_dir = "/gscratch2/fb_af1"

def set_data_loader_params(args):
    PARAMS = {
        "COMPL_LIST" : "%s/list.hetero.csv"%compl_dir,
        "HOMO_LIST" : "%s/list.homo.csv"%compl_dir,
        "NEGATIVE_LIST" : "%s/list.negative.csv"%compl_dir,
        #"PDB_LIST"   : "%s/list_v02.csv"%base_dir,
        "PDB_LIST"    : "/gscratch2/PDB-2021AUG02/list_v02.csv",
        "FB_LIST"    : "%s/list_b1-3.csv"%fb_dir,
        #"VAL_PDB"    : "%s/val/xaa"%base_dir,
        "VAL_PDB"   : "/gscratch2/PDB_val/xaa",
        "VAL_COMPL"  : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"    : "%s/val_lists/xaa.neg"%compl_dir,
        "PDB_DIR"    : base_dir,
        "FB_DIR"     : fb_dir,
        "COMPL_DIR"  : compl_dir,
        "MINTPLT" : 0,
        "MAXTPLT" : 5,
        "MINSEQ"  : 1,
        "MAXSEQ"  : 1024,
        "MAXLAT"  : 128, 
        "CROP"    : 256,
        "DATCUT"  : "2020-Apr-30",
        "RESCUT"  : 5.0,
        "BLOCKCUT": 5,
        "PLDDTCUT": 70.0,
        "SCCUT"   : 90.0,
        "ROWS"    : 1,
        "SEQID"   : 95.0,
        "MAXCYCLE": 4
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Down-sample given MSA by randomly delete blocks of sequences
    Input: MSA/Insertion having shape (N, L)
    output: new MSA/Insertion with block deletion (N', L)
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    # Get statistics from clustering results (clustering extra sequences with seed sequences)
    csum = torch.zeros(
        (N_seq, N_res, data.shape[-1]), device=data.device
    ).scatter_add(
        0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float()
    )
    return csum

def MSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, L_s=[]):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    N, L = msa.shape
    
    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Nclust sequences will be selected randomly as a seed MSA (aka latent MSA)
    # - First sequence is always query sequence
    # - the rest of sequences are selected randomly
    Nclust = min(N, params['MAXLAT'])
    
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample = torch.randperm(N-1, device=msa.device)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone())
       
        ## get extra sequenes
        if N - Nclust >= params['MAXSEQ']: # there are enough extra sequences
            Nextra = params['MAXSEQ']
            msa_extra = torch.cat((msa_masked[:1,:], msa[1:,:][sample[Nclust-1:]]), dim=0) 
            ins_extra = torch.cat((ins_clust[:1,:], ins[1:,:][sample[Nclust-1:]]), dim=0)
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
            extra_mask[0] = mask_pos[0]
        elif N - Nclust < 1: # no extra sequences, use all masked seed sequence as extra one
            Nextra = Nclust
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        else: # it has extra sequences, but not enough to maxseq. Use mixture of seed (except query) & extra
            Nextra = min(N, params['MAXSEQ'])
            msa_add = msa[1:,:][sample[Nclust-1:]]
            ins_add = ins[1:,:][sample[Nclust-1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)
        N_extra_pool = msa_extra.shape[0]
        
        # 1. one_hot encoded aatype: msa_clust_onehot
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22) # (N, L, 22)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20) # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20) 
        # get number of identical tokens for each pair of sequences (extra vs seed)
        agreement = torch.matmul(
            (count_extra[:,:,None]*msa_extra_onehot).float().view(N_extra_pool, -1), 
            (count_clust[:,:,None]*msa_clust_onehot).float().view(Nclust, -1).T) # (N_extra_pool, Nclust)
        assignment = torch.argmax(agreement, dim=-1) # map each extra seq to the closest seed seq

        # 2. cluster profile -- ignore masked token when calculate profiles
        count_extra = ~extra_mask # only consider non-masked tokens in extra seqs
        count_clust = ~mask_pos # only consider non-masked tokens in seed seqs
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_onehot
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L) # 
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]

        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        
        # seed MSA features (one-hot aa, cluster profile, ins statistics, terminal info)
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

        # extra MSA features (one-hot aa, insertion, terminal info)
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, npick_global=None, pick_top=True, random_noise=5.0):
    if npick_global == None:
        npick_global=max(npick, 1)
    seqID_cut = params['SEQID']

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ - return fake templ
        xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npick_global,qlen,1,1) + torch.rand(npick_global,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((npick_global, qlen), 20).long(), num_classes=21).float() # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global,qlen,27), False)
        return xyz, t1d, mask_t
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[tplt_valid_idx]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates -- return fake templ
        xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npick_global,qlen,1,1) + torch.rand(npick_global,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((npick_global, qlen), 20).long(), num_classes=21).float() # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global,qlen,27), False)
        return xyz, t1d, mask_t

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npick_global,qlen,1,1) + torch.rand(1,qlen,1,3)*random_noise
    mask_t = torch.full((npick_global,qlen,27),False) # True for valid atom, False for missing atom
    t1d = torch.full((npick_global, qlen), 20).long()
    t1d_val = torch.zeros((npick_global, qlen)).float()

    for i,nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt['qmap'][0,:,1]==tplt_idx)[0]
        pos = tplt['qmap'][0,sel,0] + offset
        xyz[i,pos,:14] = tplt['xyz'][0,sel]
        mask_t[i,pos,:14] = tplt['mask'][0,sel].bool()
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence
        xyz[i] = center_and_realign_missing(xyz[i], mask_t[i])

    t1d = torch.nn.functional.one_hot(t1d, num_classes=21).float()
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    return xyz, t1d, mask_t

def get_train_valid_set(params, OFFSET=1000000):
    # read validation IDs for PDB set
    val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
    val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
    val_neg_ids = set([int(l)+OFFSET for l in open(params['VAL_NEG']).readlines()])
    
    # read homo-oligomer list
    homo = {}
    with open(params['HOMO_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read pdbA, pdbB, bioA, opA, bioB, opB
        rows = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in reader]
    for r in rows:
        if r[0] in homo.keys():
            homo[r[0]].append(r[1:])
        else:
            homo[r[0]] = [r[1:]]

    # read & clean list.csv
    with open(params['PDB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4]), int(r[-1].strip())] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    # compile training and validation sets
    val_hash = list()
    train_pdb = {}
    valid_pdb = {}
    valid_homo = {}
    for r in rows:
        if r[2] in val_pdb_ids:
            val_hash.append(r[1])
            if r[2] in valid_pdb.keys():
                valid_pdb[r[2]].append((r[:2], r[-1]))
            else:
                valid_pdb[r[2]] = [(r[:2], r[-1])]
            #
            if r[0] in homo:
                if r[2] in valid_homo.keys():
                    valid_homo[r[2]].append((r[:2], r[-1]))
                else:
                    valid_homo[r[2]] = [(r[:2], r[-1])]
        else:
            if r[2] in train_pdb.keys():
                train_pdb[r[2]].append((r[:2], r[-1]))
            else:
                train_pdb[r[2]] = [(r[:2], r[-1])]
    val_hash = set(val_hash)
    
    # compile facebook model sets
    with open(params['FB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                 if float(r[1]) > 80.0 and # overall quality should be better than pLDDT 80
                 len(r[-1].strip()) > 200] # it should have at least 200 residues
    fb = {}
    for r in rows:
        if r[2] in fb.keys():
            fb[r[2]].append((r[:2], r[-1]))
        else:
            fb[r[2]] = [(r[:2], r[-1])]
    
    # compile complex sets
    with open(params['COMPL_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read complex_pdb, pMSA_hash, complex_cluster, length, taxID, assembly (bioA,opA,bioB,opB)
        rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')], r[6] , [int(r[7]), int(r[8]), int(r[9]), int(r[10])]] for r in reader
                if float(r[2]) <= params['RESCUT'] and
                parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

    train_compl = {}
    valid_compl = {}
    for r in rows:
        if r[2] in val_compl_ids:
            if r[2] in valid_compl.keys():
                valid_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1])) # ((pdb, hash), length, taxID, assembly, negative?)
            else:
                valid_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]
        else:
            # if subunits are included in PDB validation set, exclude them from training
            hashA, hashB = r[1].split('_')
            if hashA in val_hash:
                continue
            if hashB in val_hash:
                continue
            if r[2] in train_compl.keys():
                train_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1]))
            else:
                train_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]

    # compile negative examples
    # remove pairs if any of the subunits are included in validation set
    with open(params['NEGATIVE_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
        rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    train_neg = {}
    valid_neg = {}
    for r in rows:
        if r[2] in val_neg_ids:
            if r[2] in valid_neg.keys():
                valid_neg[r[2]].append((r[:2], r[-2], r[-1], []))
            else:
                valid_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]
        else:
            hashA, hashB = r[1].split('_')
            if hashA in val_hash:
                continue
            if hashB in val_hash:
                continue
            if r[2] in train_neg.keys():
                train_neg[r[2]].append((r[:2], r[-2], r[-1], []))
            else:
                train_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]
    
    # Get average chain length in each cluster and calculate weights
    pdb_IDs = list(train_pdb.keys())
    fb_IDs = list(fb.keys())
    compl_IDs = list(train_compl.keys())
    neg_IDs = list(train_neg.keys())
    #
    #pdb_weights = np.array([train_pdb[key][0][1] for key in pdb_IDs])
    #pdb_weights = (1/512.)*np.clip(pdb_weights, 256, 512)
    #fb_weights = np.array([fb[key][0][1] for key in fb_IDs])
    #fb_weights = (1/512.)*np.clip(fb_weights, 256, 512)
    #compl_weights = np.array([sum(train_compl[key][0][1]) for key in compl_IDs])
    #compl_weights = (1/512.)*np.clip(compl_weights, 256, 512)
    #neg_weights = np.array([sum(train_neg[key][0][1]) for key in neg_IDs])
    #neg_weights = (1/512.)*np.clip(neg_weights, 256, 512)

    pdb_weights = list()
    fb_weights = list()
    compl_weights = list()
    neg_weights = list()
    for key in pdb_IDs:
        plen = sum([plen for _, plen in train_pdb[key]]) // len(train_pdb[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        pdb_weights.append(w)
    
    for key in fb_IDs:
        plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        fb_weights.append(w)
    
    for key in compl_IDs:
        plen = sum([sum(plen) for _, plen, _, _ in train_compl[key]]) // len(train_compl[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        compl_weights.append(w)
    
    for key in neg_IDs:
        plen = sum([sum(plen) for _, plen, _, _ in train_neg[key]]) // len(train_neg[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        neg_weights.append(w)

    return (pdb_IDs, torch.tensor(pdb_weights).float(), train_pdb), \
           (fb_IDs, torch.tensor(fb_weights).float(), fb), \
           (compl_IDs, torch.tensor(compl_weights).float(), train_compl), \
           (neg_IDs, torch.tensor(neg_weights).float(), train_neg),\
           valid_pdb, valid_homo, valid_compl, valid_neg, homo

# slice long chains
def get_crop(l, mask, device, crop_size, unclamp=False):

    sel = torch.arange(l,device=device)
    if l <= crop_size:
        return sel
    
    size = crop_size

    mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]

    if unclamp: # bias it toward N-term.. (follow what AF did.. but don't know why)
        x = np.random.randint(len(exists)) + 1
        res_idx = exists[torch.randperm(x)[0]].item()
    else:
        res_idx = exists[torch.randperm(len(exists))[0]].item()
    lower_bound = max(0, res_idx-size+1)
    upper_bound = min(l-size, res_idx+1)
    start = np.random.randint(lower_bound, upper_bound)
    return sel[start:start+size]

def get_complex_crop(len_s, mask, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)
    
    n_added = 0
    n_remaining = sum(len_s)
    preset = 0
    sel_s = list()
    for k in range(len(len_s)):
        n_remaining -= len_s[k]
        crop_max = min(params['CROP']-n_added, len_s[k])
        crop_min = min(len_s[k], max(1, params['CROP'] - n_added - n_remaining))
        
        if k == 0:
            crop_max = min(crop_max, params['CROP']-5)
        crop_size = np.random.randint(crop_min, crop_max+1)
        n_added += crop_size
        
        mask_chain = ~(mask[preset:preset+len_s[k],:3].sum(dim=-1) < 3.0)
        exists = mask_chain.nonzero()[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        lower_bound = max(0, res_idx - crop_size + 1)
        upper_bound = min(len_s[k]-crop_size, res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + preset
        sel_s.append(sel[start:start+crop_size])
        preset += len_s[k]
    return torch.cat(sel_s)

def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-6):
    device = xyz.device
    
    # get interface residue
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1]) 
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        print ("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps
    cond = mask[:,1]*mask[cnt_idx,1]
    dist[~cond] = 999999.9
    _, idx = torch.topk(dist, params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel

# merge msa & insertion statistics of two proteins having different taxID
def merge_a3m_hetero(a3mA, a3mB, L_s, orig={}):
    # merge msa
    if 'msa' in orig:
        msa = [orig['msa']]
    else:
        query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)
        msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,L_s[1]), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)
    
    # merge ins
    if 'ins' in orig:
        ins = [orig['ins']]
    else:
        query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
        ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,L_s[1]), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)
    return {'msa': msa, 'ins': ins}

# merge msa & insertion statistics of units in homo-oligomers
#def merge_a3m_homo(msa_orig, ins_orig, nmer):
#    N, L = msa_orig.shape[:2]
#    msa = torch.cat([msa_orig for imer in range(nmer)], dim=1)
#    ins = torch.cat([ins_orig for imer in range(nmer)], dim=1)
#    return msa, ins
#def merge_a3m_homo(msa_orig, ins_orig, nmer):
#    N, L = msa_orig.shape[:2]
#    msa = torch.full((1+(N-1)*nmer, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
#    ins = torch.full((1+(N-1)*nmer, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)
#    start=0
#    start2 = 1
#    for i_c in range(nmer):
#        msa[0, start:start+L] = msa_orig[0] 
#        msa[start2:start2+(N-1), start:start+L] = msa_orig[1:]
#        ins[0, start:start+L] = ins_orig[0]
#        ins[start2:start2+(N-1), start:start+L] = ins_orig[1:]
#        start += L
#        start2 += (N-1)
#    return msa, ins

#fd
def merge_a3m_homo(msa_orig, ins_orig, nmer):
    N, L = msa_orig.shape[:2]
    msa = torch.full((2*N-1, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
    ins = torch.full((2*N-1, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)

    msa[:N, :L] = msa_orig
    ins[:N, :L] = ins_orig
    start = L

    for i_c in range(1,nmer):
        msa[0, start:start+L] = msa_orig[0] 
        msa[N:, start:start+L] = msa_orig[1:]
        ins[0, start:start+L] = ins_orig[0]
        ins[N:, start:start+L] = ins_orig[1:]
        start += L
    return msa, ins



# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True, random_noise=5.0):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t,f1d_t,mask_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = INIT_CRDS.reshape(1, 27, 3).repeat(len(idx), 1, 1)
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), 27), False)
    mask[:,:14] = pdb['mask']
    xyz = torch.nan_to_num(xyz)

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    #print ("featurize_single", ntempl, xyz_t.shape, msa_seed.shape, msa_extra.shape)
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False, 'C1'

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True, random_noise=5.0):
    L = msa_orig.shape[1]

    # msa always over 2 subunits (higher-order symms expand this)
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=[L,L])

    # get ground-truth structures
    # load metadata
    PREFIX = "%s/torch/pdb/%s/%s"%(params['PDB_DIR'],pdbid[1:3],pdbid)
    meta = torch.load(PREFIX+".pt")

    # get all possible pairs
    npairs = len(interfaces)
    xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npairs, 2*L, 1, 1)
    mask = torch.full((npairs, 2*L, 27), False)
    #print ("featurize_homo",pdbid,interfaces)
    for i_int,interface in enumerate(interfaces):
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface[0][1:3]+'/'+interface[0]+'.pt')
        xformA = meta['asmb_xform%d'%interface[1]][interface[2]]
        xformB = meta['asmb_xform%d'%interface[3]][interface[4]]
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz[i_int,:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask[i_int,:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    # detect any point symmetries
    symmgp, symmsubs = get_symmetry(xyz,mask)
    nsubs = len(symmsubs)+1

    # build full native complex (for loss calcs)
    if (symmgp != 'C1'):
        xyzfull = torch.zeros((1,nsubs*L,27,3))
        maskfull = torch.full((1,nsubs*L,27), False)
        xyzfull[0,:L] = xyz[0,:L]
        maskfull[0,:L] = mask[0,:L]
        for i in range(1,nsubs):
            xyzfull[0,i*L:(i+1)*L] = xyz[symmsubs[i-1],L:]
            maskfull[0,i*L:(i+1)*L] = mask[symmsubs[i-1],L:]
        xyz = xyzfull
        mask = maskfull

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    if ntempl < 1:
        xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    else:
        xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        # duplicate

    if (symmgp != 'C1'):
        # everything over ASU
        idx = torch.arange(L)
        chain_idx = torch.ones((L, L)).long()
        nsub = len(symmsubs)+1
    else:  # either asymmetric dimer or (usually) helical symmetry...
        # everything over 2 copies
        xyz_t = torch.cat([xyz_t, random_rot_trans(xyz_t)], dim=1)
        f1d_t = torch.cat([f1d_t]*2, dim=1)
        mask_t = torch.cat([mask_t]*2, dim=1)
        idx = torch.arange(L*2)
        idx[L:] += 100 # to let network know about chain breaks

        chain_idx = torch.zeros((2*L, 2*L)).long()
        chain_idx[:L, :L] = 1
        chain_idx[L:, L:] = 1

        nsub = 2

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # figure out crop
    if (symmgp =='C1'):
        cropsub = 2
    elif (symmgp[0]=='C'):
        cropsub = min(3, int(symmgp[1:]))
    elif (symmgp[0]=='D'):
        cropsub = min(5, 2*int(symmgp[1:]))
    else:
        cropsub = 6

    # Residue cropping
    if cropsub*L > params['CROP']:
        #if np.random.rand() < 0.5: # 50% --> interface crop
        #    spatial_crop_tgt = np.random.randint(0, npairs)
        #    crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt][0])
        #else: # 50% --> have same cropped regions across all copies
        #    crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//2, unclamp=False) # cropped region for first copy
        #    crop_idx = torch.cat((crop_idx, crop_idx+L)) # get same crops
        #    #print ("check_crop", crop_idx, crop_idx.shape)

        # fd: always use same cropped regions across all copies
        crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//cropsub, unclamp=False) # cropped region for first copy
        crop_idx_full = torch.cat([crop_idx,crop_idx+L])
        if (symmgp == 'C1'):
            crop_idx = crop_idx_full
            crop_idx_complete = crop_idx_full
        else:
            crop_idx_complete = []
            for i in range(nsub):
                crop_idx_complete.append(crop_idx+i*L)
            crop_idx_complete = torch.cat(crop_idx_complete)

        # over 2 copies
        seq = seq[:,crop_idx_full]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx_full]
        msa_seed = msa_seed[:,:,crop_idx_full]
        msa_extra = msa_extra[:,:,crop_idx_full]
        mask_msa = mask_msa[:,:,crop_idx_full]

        # over 1 copy (symmetric) or 2 copies (asymmetric)
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        mask_t = mask_t[:,crop_idx]
        idx = idx[crop_idx]
        chain_idx = chain_idx[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]
        mask_prev = mask_prev[crop_idx]

        # over >=2 copies
        xyz = xyz[:,crop_idx_complete]
        mask = mask[:,crop_idx_complete]

    #print ("featurize_homo", ntempl, xyz_t.shape, msa_seed.shape, msa_extra.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, symmgp

def get_pdb(pdbfilename, plddtfilename, item, lddtcut, sccut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    
    # update mask info with plddt (ignore sidechains if plddt < 90.0)
    #mask_lddt = np.full_like(mask, False)
    #mask_lddt[plddt > sccut] = True
    #mask_lddt[:,:5] = True
    #mask = np.logical_and(mask, mask_lddt)
    mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx': torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item, max_seq=8000):
    msa,ins = parse_a3m(a3mfilename, max_seq=max_seq)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5):
    # load MSA, PDB, template info
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt')
   
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    if item[0] in homo: # Target is homo-oligomer
        p_homo = np.random.rand()
        if p_homo < p_homo_cut: # model as homo-oligomer with p_homo_cut prob
            pdbid = item[0].split('_')[0]
            interfaces = homo[item[0]]
            return featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top)
        else:
            return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
    else:
        return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
    
def loader_fb(item, params, unclamp=False, random_noise=5.0):
    
    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'], params['SCCUT'])
    
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    l_orig = msa.shape[1]
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features -- None
    xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(1,l_orig,1,1) + torch.rand(1,l_orig,1,3)*random_noise
    f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=21).float() # all gaps
    conf = torch.zeros((1, l_orig, 1)).float()
    f1d_t = torch.cat((f1d_t, conf), -1)
    mask_t = torch.full((1,l_orig,27), False)
    
    idx = pdb['idx']
    xyz = INIT_CRDS.reshape(1,27,3).repeat(len(idx), 1, 1)
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), 27), False)
    mask[:,:14] = pdb['mask']

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # initial structure
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    
    #print ("loader_fb", 0, xyz_t.shape, msa_seed.shape, msa_extra.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False, 'C1'

def loader_complex(item, L_s, taxID, assem, params, negative=False, pick_top=True, random_noise=5.0):
    pdb_pair = item[0]
    pMSA_hash = item[1]
    
    msaA_id, msaB_id = pMSA_hash.split('_')
    a3m = {}
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        a3m = get_msa(pMSA_fn, pMSA_hash)
    # read MSA for each subunit & merge them
    a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
    a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
    a3mA = get_msa(a3mA_fn, msaA_id, max_seq=params['MAXSEQ']*2)
    a3mB = get_msa(a3mB_fn, msaB_id, max_seq=params['MAXSEQ']*2)
    a3m = merge_a3m_hetero(a3mA, a3mB, L_s, orig=a3m)

    # get MSA features
    msa = a3m['msa'].long()
    if negative: # Qian's paired MSA for true-pairs have no insertions... (ignore insertion to avoid any weird bias..) 
        ins = torch.zeros_like(msa)
    else:
        ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=L_s)

    # read template info
    tpltA_fn = params['PDB_DIR'] + '/torch/hhr/' + msaA_id[:3] + '/' + msaA_id + '.pt'
    tpltB_fn = params['PDB_DIR'] + '/torch/hhr/' + msaB_id[:3] + '/' + msaB_id + '.pt'
    tpltA = torch.load(tpltA_fn)
    tpltB = torch.load(tpltB_fn)

    ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    ntemplB = np.random.randint(0, params['MAXTPLT']+1-ntemplA)
    xyz_t_A, f1d_t_A, mask_t_A = TemplFeaturize(tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_B, f1d_t_B, mask_t_B = TemplFeaturize(tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
    mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt')
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt')
    
    if len(assem) > 0:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt')

        # get transform
        xformA = meta['asmb_xform%d'%assem[0]][assem[1]]
        xformB = meta['asmb_xform%d'%assem[2]][assem[3]]
        
        # apply transform
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz = INIT_CRDS.reshape(1, 27, 3).repeat(sum(L_s), 1, 1)
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask = torch.full((sum(L_s), 27), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    else:
        xyz = INIT_CRDS.reshape(1, 27, 3).repeat(sum(L_s), 1, 1)
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
        mask = torch.full((sum(L_s), 27), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)
    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 100

    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1

    # Do cropping
    if sum(L_s) > params['CROP']:
        if negative:
            sel = get_complex_crop(L_s, mask, seq.device, params)
        else:
            sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, pdb_pair)
        #
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[sel]
        mask = mask[sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
    
    #print ("loader_compl", ntempl, xyz_t.shape, msa_seed.shape, msa_extra.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative, 'C1'

class Dataset(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, homo, unclamp_cut=0.9, pick_top=True, p_homo_cut=-1.0):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.homo = homo
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.p_homo_cut = p_homo_cut

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        p_unclamp = np.random.rand()
        if p_unclamp > self.unclamp_cut:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo,
                              unclamp=True, 
                              pick_top=self.pick_top, 
                              p_homo_cut=self.p_homo_cut)
        else:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo, 
                              pick_top=self.pick_top,
                              p_homo_cut=self.p_homo_cut)
        return out

class DatasetComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(self.item_dict[ID][sel_idx][0],
                          self.item_dict[ID][sel_idx][1],
                          self.item_dict[ID][sel_idx][2],
                          self.item_dict[ID][sel_idx][3],
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative)
        return out

class DistilledDataset(data.Dataset):
    def __init__(self,
                 pdb_IDs,
                 pdb_loader,
                 pdb_dict,
                 compl_IDs,
                 compl_loader,
                 compl_dict,
                 neg_IDs,
                 neg_loader,
                 neg_dict,
                 fb_IDs,
                 fb_loader,
                 fb_dict,
                 homo,
                 params,
                 p_homo_cut=0.5):
        #
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.pdb_loader = pdb_loader
        self.compl_IDs = compl_IDs
        self.compl_loader = compl_loader
        self.compl_dict = compl_dict
        self.neg_IDs = neg_IDs
        self.neg_loader = neg_loader
        self.neg_dict = neg_dict
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.fb_loader = fb_loader
        self.homo = homo
        self.params = params
        self.unclamp_cut = 0.9
        self.p_homo_cut = p_homo_cut
        
        self.compl_inds = np.arange(len(self.compl_IDs))
        self.neg_inds = np.arange(len(self.neg_IDs))
        self.fb_inds = np.arange(len(self.fb_IDs))
        self.pdb_inds = np.arange(len(self.pdb_IDs))
    
    def __len__(self):
        return len(self.fb_inds) + len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds)

    def __getitem__(self, index):
        p_unclamp = np.random.rand()
        if index >= len(self.fb_inds) + len(self.pdb_inds) + len(self.compl_inds): # from negative set
            ID = self.neg_IDs[index-len(self.fb_inds)-len(self.pdb_inds)-len(self.compl_inds)]
            sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
            out = self.neg_loader(self.neg_dict[ID][sel_idx][0], self.neg_dict[ID][sel_idx][1], self.neg_dict[ID][sel_idx][2], self.neg_dict[ID][sel_idx][3], self.params, negative=True)

        elif index >= len(self.fb_inds) + len(self.pdb_inds): # from complex set
            ID = self.compl_IDs[index-len(self.fb_inds)-len(self.pdb_inds)]
            sel_idx = np.random.randint(0, len(self.compl_dict[ID]))
            out = self.compl_loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2], self.compl_dict[ID][sel_idx][3], self.params, negative=False)

        elif index >= len(self.fb_inds): # from PDB set
            ID = self.pdb_IDs[index-len(self.fb_inds)]
            sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, self.homo, unclamp=True, p_homo_cut=self.p_homo_cut)
            else:
                out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, self.homo, unclamp=False, p_homo_cut=self.p_homo_cut)
        else: # from FB set
            ID = self.fb_IDs[index]
            sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=True)
            else:
                out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=False)
        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(self, dataset, pdb_weights, compl_weights, neg_weights, fb_weights, num_example_per_epoch=25600, \
                 fraction_fb=0.5, fraction_compl=0.25, num_replicas=None, rank=None, replacement=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        assert num_example_per_epoch % num_replicas == 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_compl_per_epoch = int(round(num_example_per_epoch*(1.0-fraction_fb)*fraction_compl))
        self.num_neg_per_epoch = int(round(num_example_per_epoch*(1.0-fraction_fb)*fraction_compl))
        self.num_fb_per_epoch = int(round(num_example_per_epoch*(fraction_fb)))
        self.num_pdb_per_epoch = num_example_per_epoch - self.num_compl_per_epoch - self.num_neg_per_epoch - self.num_fb_per_epoch
        #print (self.num_compl_per_epoch, self.num_neg_per_epoch, self.num_fb_per_epoch, self.num_pdb_per_epoch)
        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.pdb_weights = pdb_weights
        self.compl_weights = compl_weights
        self.neg_weights = neg_weights
        self.fb_weights = fb_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # 1. subsample fb and pdb based on length
        sel_indices = torch.tensor((),dtype=int)
        if (self.num_fb_per_epoch>0):
            fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[fb_sampled]))

        if (self.num_pdb_per_epoch>0):
            pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[pdb_sampled + len(self.dataset.fb_IDs)]))

        if (self.num_compl_per_epoch>0):
            compl_sampled = torch.multinomial(self.compl_weights, self.num_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[compl_sampled + len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs)]))
        
        if (self.num_neg_per_epoch>0):
            neg_sampled = torch.multinomial(self.neg_weights, self.num_neg_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[neg_sampled + len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs) + len(self.dataset.compl_IDs)]))

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

