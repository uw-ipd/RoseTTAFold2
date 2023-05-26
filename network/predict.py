import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates, read_template_pdb, parse_pdb
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from featurizing import MSAFeaturize, MSABlockDeletion
from kinematics import xyz_to_c6d, xyz_to_t2d
from chemical import INIT_CRDS
from util_module import XYZConverter
from symmetry import symm_subunit_matrix, find_symm_subs, get_symm_map
from data_loader import merge_a3m_hetero
import json

# suppress dgl warning w/ newest pytorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold2NA")
    parser.add_argument("-inputs", help="R|Input data in format A:B:C, with\n"
         "   A = multiple sequence alignment file\n"
         "   B = hhpred hhr file\n"
         "   C = hhpred atab file\n"
         "Spaces seperate multiple inputs.  The last two arguments may be omitted\n",
         default=None, nargs='+')
    parser.add_argument("-db", help="HHpred database location", default=None)
    parser.add_argument("-prefix", help="Output prefix", type=str, default="S")
    parser.add_argument("-symm", default="C1", help="Symmetry.  IF PROVIDED, 'input' should cover the asymmetric unit")
    parser.add_argument("-model", default=None, help="Model weights")
    args = parser.parse_args()
    return args

MAX_CYCLE = 12
NMODEL = 1
NBIN = [37, 37, 37, 19]
SUBCROP = -1

MAXLAT=128
MAXSEQ=1024

MODEL_PARAM ={
        "n_extra_block": 4,
        "n_main_block": 36,
        "d_msa"           : 256 ,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 32,
        "p_drop"       : 0.0,
        }

SE3_param_full = {
        "num_layers"    : 1,
        "num_channels"  : 48,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 32,
        "l1_in_features": 2,
        "l1_out_features": 2,
        "num_edge_features": 32,
        "div": 4,
        "n_heads": 4
        }

SE3_param_topk = {
        "num_layers"    : 1,
        "num_channels"  : 128,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 2,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param_full'] = SE3_param_full
MODEL_PARAM['SE3_param_topk'] = SE3_param_topk

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

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

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

class Predictor():
    def __init__(self, model_weights, device="cuda:0"):
        # define model name
        self.model_weights = model_weights
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM
        ).to(self.device)

        could_load = self.load_model(self.model_weights)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        # from xyz to get xxxx or from xxxx to xyz
        self.l2a = util.long2alt.to(self.device)
        self.aamask = util.allatom_mask.to(self.device)
        self.lddt_bins = torch.linspace(1.0/50, 1.0, 50, device=self.device) - 1.0/100

        self.xyz_converter = XYZConverter()


    def load_model(self, model_weights):
        if not os.path.exists(model_weights):
            return False
        checkpoint = torch.load(model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True

    def predict(self, inputs, out_prefix, symm="C1", ffdb=None, n_templ=4):
        symmids,symmRs,symmmeta,symmoffset = symm_subunit_matrix(symm)
        O = symmids.shape[0]

        ###
        # pass 1, combined MSA
        Ls_blocked, Ls, msas, inss = [], [], [], []
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            a3m_i = fseq_i[0]
            msa_i, ins_i, Ls_i = parse_a3m(a3m_i)
            msa_i = torch.tensor(msa_i).long()
            ins_i = torch.tensor(ins_i).long()
            if (msa_i.shape[0] > MAXSEQ):
                idxs_tokeep = np.random.permutation(msa_i.shape[0])[:MAXSEQ]
                idxs_tokeep[0] = 0  # keep best
                msa_i = msa_i[idxs_tokeep]
                ins_i = ins_i[idxs_tokeep]

            msas.append(msa_i)
            inss.append(ins_i)
            Ls.extend(Ls_i)
            Ls_blocked.append(msa_i.shape[0])

        msa_orig = {'msa':msas[0],'ins':inss[0]}
        for i in range(1,len(Ls_blocked)):
            msa_orig = merge_a3m_hetero(msa_orig, {'msa':msas[i],'ins':inss[i]}, [sum(Ls_blocked[:i]),Ls_blocked[i]])
        msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

        ###
        # pass 2, templates
        L = sum(Ls)
        xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(n_templ,L,1,1) + torch.rand(n_templ,L,1,3)*5.0 - 2.5
        mask_t = torch.full((n_templ, L, 27), False) 
        t1d = torch.nn.functional.one_hot(torch.full((n_templ, L), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((n_templ,L,1)).float()), -1)

        maxtmpl=1
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            if (len(fseq_i) == 3):
                hhr_i,atab_i = fseq_i[1:3]
                startres,stopres = sum(Ls_blocked[:i]), sum(Ls_blocked[:(i+1)])
                xyz_t_i, t1d_i, mask_t_i = read_templates(Ls_blocked[i], ffdb, hhr_i, atab_i, n_templ=n_templ)
                ntmpl_i = xyz_t_i.shape[0]
                maxtmpl = max(maxtmpl, ntmpl_i)
                xyz_t[:ntmpl_i,startres:stopres,:,:] = xyz_t_i
                t1d[:ntmpl_i,startres:stopres,:] = t1d_i
                mask_t[:ntmpl_i,startres:stopres,:] = mask_t_i

        same_chain = torch.zeros((1,L,L), dtype=torch.bool, device=xyz_t.device)
        stopres = 0
        for i in range(1,len(Ls)):
            startres,stopres = sum(Ls[:(i-1)]), sum(Ls[:i])
            same_chain[:,startres:stopres,startres:stopres] = True
        same_chain[:,stopres:,stopres:] = True

        # template features
        xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
        mask_t = mask_t[:maxtmpl].unsqueeze(0)
        t1d = t1d[:maxtmpl].float().unsqueeze(0)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, mask_in=mask_t.reshape(-1,L,27))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3*10)

        ###
        # pass 3, symmetry
        xyz_prev = xyz_t[:,0]
        xyz_prev, symmsub = find_symm_subs(xyz_prev[:,:L],symmRs,symmmeta)

        Osub = symmsub.shape[0]
        mask_t = mask_t.repeat(1,1,Osub,1)
        alpha_t = alpha_t.repeat(1,1,Osub,1)
        mask_prev = mask_t[:,0]
        xyz_t = xyz_t.repeat(1,1,Osub,1,1)
        t1d = t1d.repeat(1,1,Osub,1)

        # symmetrize msa
        effL = Osub*L
        if (Osub>1):
            msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub)

        # index
        idx_pdb = torch.arange(Osub*L)[None,:]

        same_chain = torch.zeros((1,Osub*L,Osub*L)).long()
        i_start = 0
        for o_i in range(Osub):
            for li in Ls:
                i_stop = i_start + li
                idx_pdb[:,i_stop:] += 100
                same_chain[:,i_start:i_stop,i_start:i_stop] = 1
                i_start = i_stop

        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float()*same_chain.float()[:,None] # (ignore inter-chain region)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)


        self.model.eval()
        for i_trial in range(NMODEL):
            #if os.path.exists("%s_%02d_init.pdb"%(out_prefix, i_trial)):
            #    continue
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            self.run_prediction(
                msa_orig, ins_orig, t1d, t2d, xyz_t[:,:,:,1], alpha_t, mask_t_2d, xyz_prev, mask_prev, same_chain, idx_pdb,
                symmids, symmsub, symmRs, symmmeta,  Ls, "%s_%02d"%(out_prefix, i_trial))
            max_mem = torch.cuda.max_memory_allocated()/1e9
            print ("Memory used:", max_mem, "/ Time: %.2f sec"%(time.time()-start_time))
            torch.cuda.empty_cache()

    def run_prediction(self, msa_orig, ins_orig, t1d, t2d, xyz_t, alpha_t, mask_t, xyz_prev, mask_prev, same_chain, idx_pdb, symmids, symmsub, symmRs, symmmeta, L_s, out_prefix):
        self.xyz_converter = self.xyz_converter.to(self.device)

        with torch.no_grad():
            msa = msa_orig.long().to(self.device) # (N, L)
            ins = ins_orig.long().to(self.device)

            print ("Input size", msa.shape, ins.shape)
            N, L = msa.shape[:2]
            O = symmids.shape[0]
            Osub = symmsub.shape[0]
            Lasu = L//Osub

            B = 1
            #
            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t.to(self.device)
            mask_t = mask_t.to(self.device)
            alpha_t = alpha_t.to(self.device)
            xyz_prev = xyz_prev.to(self.device)
            mask_prev = mask_prev.to(self.device)
            same_chain = same_chain.to(self.device)
            symmids = symmids.to(self.device)
            symmsub = symmsub.to(self.device)
            symmRs = symmRs.to(self.device)

            subsymms, _ = symmmeta
            for i in range(len(subsymms)):
                subsymms[i] = subsymms[i].to(self.device)

            msa_prev=None
            pair_prev=None
            state_prev=None
            mask_recycle = mask_prev[:,:,:3].bool().all(dim=-1)
            mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
            mask_recycle = same_chain.float()*mask_recycle.float()

            best_lddt = torch.tensor([-1.0], device=self.device)
            best_xyz = None
            best_logit = None
            best_aa = None
            best_pae = None
            for i_cycle in range(MAX_CYCLE):
                seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                    msa, ins, p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': 1})

                seq = seq.unsqueeze(0)
                msa_seed = msa_seed.unsqueeze(0)
                msa_extra = msa_extra.unsqueeze(0)

                with torch.cuda.amp.autocast(True):
                    logit_s, logit_aa_s, _, logits_pae, p_bind, xyz_prev, alpha, symmsub, pred_lddt, msa_prev, pair_prev, state_prev = self.model(
                                                               msa_seed.half(), msa_extra.half(),
                                                               seq, xyz_prev, 
                                                               idx_pdb,
                                                               t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                               alpha_t=alpha_t, mask_t=mask_t,
                                                               same_chain=same_chain,
                                                               msa_prev=msa_prev,
                                                               pair_prev=pair_prev,
                                                               state_prev=state_prev,
                                                               p2p_crop=SUBCROP,
                                                               #topk_crop=SUBCROP,
                                                               mask_recycle=mask_recycle,
                                                               symmids=symmids,
                                                               symmsub=symmsub,
                                                               symmRs=symmRs,
                                                               symmmeta=symmmeta )

                    alpha = alpha[-1]
                    xyz_prev = xyz_prev[-1]
                    _, xyz_prev = self.xyz_converter.compute_all_atom(seq, xyz_prev, alpha)
                    mask_recycle=None

                pred_lddt = nn.Softmax(dim=1)(pred_lddt) * self.lddt_bins[None,:,None]
                pred_lddt = pred_lddt.sum(dim=1)
                pae = pae_unbin(logits_pae)
                print ("RECYCLE", i_cycle, pred_lddt.mean(), pae.mean(), best_lddt.mean())
                #util.writepdb("%s_cycle_%02d.pdb"%(out_prefix, i_cycle), xyz_prev[0], seq[0], L_s, bfacts=100*pred_lddt[0])

                logit_s = [l.cpu() for l in logit_s]
                logit_aa_s = [l.cpu() for l in logit_aa_s]

                torch.cuda.empty_cache()
                if pred_lddt.mean() < best_lddt.mean():
                    continue
                
                best_xyz = xyz_prev.float().cpu()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.cpu()
                best_pae = pae.float().cpu()

            prob_s = list()
            for logit in best_logit:
                prob = self.active_fn(logit.float()) # distogram
                prob_s.append(prob)

        # full complex
        symmRs = symmRs.cpu()
        best_xyzfull = torch.zeros( (B,O*Lasu,27,3),device=best_xyz.device )
        best_xyzfull[:,:Lasu] = best_xyz[:,:Lasu]
        seq_full = torch.zeros( (B,O*Lasu),dtype=seq.dtype, device=seq.device )
        seq_full[:,:Lasu] = seq[:,:Lasu]
        best_lddtfull = torch.zeros( (B,O*Lasu),device=best_lddt.device )
        best_lddtfull[:,:Lasu] = best_lddt[:,:Lasu]
        for i in range(1,O):
            best_xyzfull[:,(i*Lasu):((i+1)*Lasu)] = torch.einsum('ij,braj->brai', symmRs[i], best_xyz[:,:Lasu])
            seq_full[:,(i*Lasu):((i+1)*Lasu)] = seq[:,:Lasu]
            best_lddtfull[:,(i*Lasu):((i+1)*Lasu)] = best_lddt[:,:Lasu]

        outdata = {}

        # RMS
        #monomer_rms, complex_rms, best_xyzfull = calc_symm_rmsd(best_xyzfull, native, O)
        #outdata['monomer_rms'] = monomer_rms.item()
        #outdata['complex_rms'] = complex_rms.item()
        outdata['mean_plddt'] = best_lddt.mean().item()
        for i in range(O):
            outdata['pae_chain0_'+str(i)] = 0.5 * (best_pae[:,0:Lasu,i*Lasu:(i+1)*Lasu].mean() + best_pae[:,i*Lasu:(i+1)*Lasu,0:Lasu].mean()).item()

        with open("%s.json"%(out_prefix), "w") as outfile:
            json.dump(outdata, outfile, indent=4)

        util.writepdb("%s_pred.pdb"%(out_prefix), best_xyzfull[0], seq_full[0], L_s, bfacts=100*best_lddtfull[0])

        prob_s = [prob.permute(0,2,3,1).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16), \
                            lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16))



if __name__ == "__main__":
    args = get_args()

    if (args.db is not None):
        FFDB = args.db
        FFindexDB = namedtuple("FFindexDB", "index, data")
        ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                         read_data(FFDB+'_pdb.ffdata'))
    else:
        ffdb = None

    if (torch.cuda.is_available()):
        print ("Running on GPU")
        pred = Predictor(args.model, torch.device("cuda:0"))
    else:
        print ("Running on CPU")
        pred = Predictor(args.model, torch.device("cpu"))

    pred.predict(inputs=args.inputs, out_prefix=args.prefix, symm=args.symm, ffdb=ffdb)

