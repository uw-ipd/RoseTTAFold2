import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from parsers import read_template_pdb, read_multichain_template_pdb
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from featurizing import MSAFeaturize, MSABlockDeletion
from kinematics import xyz_to_c6d, xyz_to_t2d
from util_module import XYZConverter
import glob
from pathlib import Path

MAX_CYCLE = 12
NREPLICATES = 1
NBIN = [37, 37, 37, 19]

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

# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]

# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

class Predictor():
    def __init__(self, model_name, device="cuda:0"):
        self.model_name = model_name
        if (self.model_name[0] != '/'):
            self.model_name = os.path.dirname(os.path.abspath(__file__))+'/'+self.model_name
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
        ).to(self.device)

        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        # from xyz to get xxxx or from xxxx to xyz
        self.l2a = util.long2alt.to(self.device)
        self.aamask = util.allatom_mask.to(self.device)
        self.lddt_bins = torch.linspace(1.0/50, 1.0, 50, device=self.device) - 1.0/100

        self.xyz_converter = XYZConverter().to(self.device)

    def load_model(self, model_name):
        chk_fn = model_name
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def predict(self, pdb_fn, out_prefix, out_path, target_chain='B', use_init_crd=False, from_scratch=False, cyclic=False ):
        msa_orig, ins_orig, L_s, xyz_t, mask_t, t1d, dslf, is_tgt = read_multichain_template_pdb(
            pdb_fn, target_chain=target_chain, templ_from_nontgt=(not from_scratch),
        )
        L1 = L_s[0]
        N,L = msa_orig.shape

        #
        idx_pdb = torch.arange(L).long().view(1, L)
        idx_pdb[:, L1:] += 100
        idx_pdb = idx_pdb.to(self.device)

        same_chain = torch.zeros((1,L,L), dtype=torch.bool, device=self.device)
        #same_chain[:,:L1,:L1] = True
        #same_chain[:,L1:,L1:] = True
        same_chain[...]=True

        # template features
        xyz_t = xyz_t.float().unsqueeze(0).to(self.device)
        mask_t = mask_t.unsqueeze(0).to(self.device)
        t1d = t1d.float().unsqueeze(0).to(self.device)
        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None]*same_chain[:,None,:,:] # (B, T, L, L)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        cyclize_reses = torch.zeros(L, dtype=torch.bool)
        if cyclic:
            cyclize_reses = ~is_tgt
        cyclize_reses = cyclize_reses.to(self.device)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, mask_in=mask_t.reshape(-1,L,27))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        # prepare initial coordinates
        xyz_prev = xyz_t[:,0]
        mask_prev = mask_t[:,0]

        self.model.eval()
        for i_trial in range(NREPLICATES):
            self.run_prediction(
                msa_orig, ins_orig, t1d, t2d, xyz_t[:,:,:,1], alpha_t, mask_t_2d, xyz_prev, mask_prev, same_chain, idx_pdb, cyclize_reses, L_s,
                "%s_%02d"%(out_prefix, i_trial),out_path)
            torch.cuda.empty_cache()

    def run_prediction(self, msa_orig, ins_orig, t1d, t2d, xyz_t, alpha_t, mask_t, xyz_prev, mask_prev, same_chain, idx_pdb, cyclize_reses, L_s, out_prefix,out_path):
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if msa_orig.shape[0] > 10000:
                msa, ins = MSABlockDeletion(msa_orig, ins_orig)
                msa = torch.tensor(msa).long().to(self.device) # (N, L)
                ins = torch.tensor(ins).long().to(self.device)
            else:
                msa = msa_orig.long().to(self.device) # (N, L)
                ins = ins_orig.long().to(self.device)

            print ("Input size", msa.shape, ins.shape)
            self.write_pdb(msa_orig[0], xyz_prev[0], prefix="%s_tmpl"%(out_prefix))

            N, L = msa.shape[:2]
            L1 = L_s[0]

            msa_prev = None
            pair_prev = None
            state_prev = None

            mask_recycle = mask_prev[:,:,:3].bool().all(dim=-1)
            mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
            mask_recycle = same_chain.float()*mask_recycle.float()
            

            best_lddt = torch.tensor([-1.0], device=self.device)
            best_xyz = None
            best_logit = None
            best_aa = None

#            print ("           PAE t/t PAE p/t PAE p/p   plddt    best")
 
            for i_cycle in range(MAX_CYCLE):
                seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                    msa, ins, p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': 1})

                seq = seq.unsqueeze(0)
                msa_seed = msa_seed.unsqueeze(0)
                msa_extra = msa_extra.unsqueeze(0)

                with torch.cuda.amp.autocast(False):
                    logit_s, logit_aa_s, _, logits_pae, _, xyz_prev, alpha, _, pred_lddt, msa_prev, pair_prev, state_prev = self.model(
                       msa_seed, msa_extra,
                       seq, xyz_prev, 
                       idx_pdb,
                       t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                       alpha_t=alpha_t, mask_t=mask_t,
                       same_chain=same_chain,
                       msa_prev=msa_prev,
                       pair_prev=pair_prev,
                       state_prev=state_prev,
                       mask_recycle=mask_recycle, 
                       cyclize_reses=cyclize_reses
                    )

                alpha = alpha[-1]
                xyz_prev = xyz_prev[-1]
                _, xyz_prev = self.xyz_converter.compute_all_atom(seq, xyz_prev, alpha)

                pred_lddt = lddt_unbin(pred_lddt)
                pae = pae_unbin(logits_pae)
                pae_pp = pae[:,:L1,:L1].mean()
                pae_pt = 0.5*(pae[:,:L1,L1:].mean() + pae[:,L1:,:L1].mean())
                pae_tt = pae[:,L1:,L1:].mean()

#                if i_cycle == MAX_CYCLE-1:
#                    out_file = f'{out_path}/scores.sc'
#                    with open(out_file,'a') as f:
#
#                        f.write("%s RECYCLE %2d %7.3f %7.3f %7.3f %7.3f %7.3f \n"%(
#                            out_prefix,
#                            i_cycle, 
#                            pae_tt, 
#                            pae_pp, 
#                            pae_pt, 
#                            pred_lddt.mean().cpu().numpy(), 
#                            best_lddt.mean().cpu().numpy()
#                        ) )
#
#                    self.write_pdb(seq[0], xyz_prev[0], Bfacts=pred_lddt[0], prefix="%s_cycle_%02d"%(out_prefix, i_cycle),chain_split=None)
            
                if pred_lddt.mean() < best_lddt.mean():
                    continue

                best_xyz = xyz_prev.clone()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.clone()
                best_pae = pae.clone()
                best_pae_pp = pae_pp.clone()
                best_pae_pt = pae_pt.clone()
                best_pae_tt = pae_tt.clone()

                best_score = pred_lddt

            out_file = f'{out_path}/scores.sc'
            new_line='\n'
            with open(out_file,'a') as f:
                f.write(f"{out_prefix},{best_pae_tt:.3f},{best_pae_pt:.3f},{best_pae_pp:.3f},{best_lddt.mean().cpu().numpy():.3f}{new_line}")

            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob_s.append(prob)
        
        end = time.time()

        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        self.write_pdb(seq[0], best_xyz[0], Bfacts=best_lddt[0], prefix="%s_best"%(out_prefix),chain_split=L1)
        prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        #np.savez_compressed(
        #    "%s.npz"%(out_prefix), 
        #        dist=prob_s[0].astype(np.float16), \
        #        omega=prob_s[1].astype(np.float16),\
        #        theta=prob_s[2].astype(np.float16),\
        #        phi=prob_s[3].astype(np.float16),\
        #        lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),\
        #        pae=best_pae[0].detach().cpu().numpy().astype(np.float16),\
        #)

        max_mem = torch.cuda.max_memory_allocated()/1e9
        print ("max mem", max_mem)
        print ("runtime", end-start)

    def write_pdb(self, seq, atoms, idx_pdb=None, Bfacts=None, prefix=None,chain_split=None):
        L = len(seq)
        if idx_pdb == None:
            idx_pdb = np.arange(L)
        if chain_split == None:
            chain_id = ['A'] * L
        else:
            chain_id = ['A']  * chain_split + ['B'] * (L - chain_split)

        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                atms = util.aa2long[s][:14]
                for j,atm_j in enumerate(atms):
                    if (atm_j is not None):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, atm_j, util.num2aa[s], 
                            chain_id[i], idx_pdb[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                            1.0, Bfacts[i] ) )
                        ctr += 1
                    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Validate designed complex with RoseTTAFold2")
    parser.add_argument('-pdb_dir', required=True,
                        help="Directory with designed complex structure file in PDB format. The residue number should be continuous starting from 1 to L_complex.")
    parser.add_argument('-out_path', required=True,
                        help="prefix for output file. The outputs will be [prefix]_??_init.pdb and [prefix]_??.npz")
    parser.add_argument('-target_chain', default='B',
                        help="The chain corresponds to target proteins. If it has two chains (e.g. chain B & C), please provide it as BC")
    parser.add_argument("-model_name", default="weights/RF2_jan24.pt", required=False, 
                        help="Prefix for model. The model [model_name] will be used. [weights/RF2_jan24.pt]")
    parser.add_argument("-from_scratch", default=False, action='store_true',
                        help="Build non-target chain from scratch?")
    parser.add_argument("-cyclic", default=False, action='store_true',
                        help="Is the non-target chain cyclic?")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    
    all_pdbs = glob.iglob(f'{args.pdb_dir}/*.pdb',recursive=True)
    for pdb_fn in all_pdbs:
        out_prefix = f'{args.out_path}/{Path(pdb_fn).stem}'
        print (pdb_fn,out_prefix)
        pred = Predictor(model_name=args.model_name)
        pred.predict(pdb_fn, out_prefix, args.out_path, target_chain=args.target_chain, from_scratch=args.from_scratch, cyclic=args.cyclic)
