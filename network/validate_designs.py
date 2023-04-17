import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates, read_template_pdb, parse_pdb, parse_pdb_w_seq
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from featurizing import MSAFeaturize, MSABlockDeletion
from kinematics import xyz_to_c6d, xyz_to_t2d
from chemical import INIT_CRDS
from util_module import XYZConverter
from symmetry import symm_subunit_matrix, find_symm_subs, get_symm_map
import json

MAX_CYCLE = 20
NMODEL = 1
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

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)


def calc_rmsd(pred, true):
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    B, L, Natm = pred.shape[:3]

    # center to centroid
    pred = pred[:,:,1,:]
    true = true[:,:,1,:]
    pred = pred - centroid(pred)
    true = true - centroid(true)

    # Computation of the covariance matrix
    C = torch.matmul(pred.permute(0,2,1), true)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)

    # Rotate pred
    rpred = torch.matmul(pred, U) # (IB, L*3, 3)

    # get RMS
    rms = rmsd(rpred, true).reshape(B)
    return rms, U

def calc_symm_rmsd(pred, true, O=1):
    B, L = pred.shape[:2]
    Lasu = L//O

    monomer_rms, U = calc_rmsd(pred[:,:Lasu], true[:,:Lasu])

    rpred = torch.matmul(pred, U)

    shufrpred = []
    used = torch.zeros(O,dtype=torch.bool)
    for i in range(O):
        com_i = true[:,i*Lasu:(i+1)*Lasu,1,:].mean(dim=-2)
        min_di, argmin_di = 999.0, -1
        for j in range(O):
            if (used[j]):
                continue
            com_j = rpred[:,j*Lasu:(j+1)*Lasu,1,:].mean(dim=-2)
            d_ij = torch.linalg.norm(com_j-com_i)
            if (d_ij < min_di):
                min_di = d_ij
                argmin_di = j
        used[argmin_di]=True
        shufrpred.append(rpred[:,argmin_di*Lasu:(argmin_di+1)*Lasu])

    shufrpred = torch.cat(shufrpred, dim=1)
    complex_rms, _ = calc_rmsd(shufrpred, true)

    return monomer_rms, complex_rms, shufrpred

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
    def __init__(self, model_dir=None, device="cuda:0"):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "BFF"
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM
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


    def load_model(self, model_name, suffix='last'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True

    def predict(self, pdb_fn, symm="C1"):
        out_prefix = os.path.basename(pdb_fn)
        out_prefix = os.path.splitext(out_prefix)[0]
        #print (out_prefix)

        native,mask,_,seq = parse_pdb_w_seq(pdb_fn)
        native = torch.tensor(native).unsqueeze(0)

        symmids,symmRs,symmmeta,symmoffset = symm_subunit_matrix(symm)

        O = symmids.shape[0]
        L = seq.shape[0]//O

        msa_orig = torch.tensor(seq[None,:L]).repeat(2,1)
        ins_orig = torch.zeros_like(msa_orig)
        N = msa_orig.shape[0]

        # dummy template
        xyz_t_baseline = INIT_CRDS.reshape(1,1,27,3).repeat(1,L,1,1) + torch.rand(1,L,1,3)*5.0 + symmoffset*L**(1/3)
        xyz_t = xyz_t_baseline
        mask_t = torch.full((1, L, 27), False) 
        t1d = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,L,1)).float()), -1)

        # find contacting subunits and symmetrize
        xyz_t, symmsub = find_symm_subs(xyz_t[:,:L],symmRs,symmmeta)

        Osub = symmsub.shape[0]
        mask_t = mask_t.repeat(1,Osub,1)
        t1d = t1d.repeat(1,Osub,1)

        # symmetrize msa
        effL = Osub*L
        if (Osub>1):
            msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub)

        # index
        idx_pdb = torch.arange(Osub*L)[None,:]

        same_chain = torch.zeros((1,Osub*L,Osub*L), device=self.device).long()
        for o_i in range(Osub):
            i = symmsub[o_i]
            same_chain[:,o_i*L:(i+1)*L,o_i*L:(o_i+1)*L] = 1
            idx_pdb[:,o_i*L:(i+1)*L] += 100*o_i

        # template features
        xyz_t = xyz_t.float().unsqueeze(0).to(self.device)
        mask_t = mask_t.unsqueeze(0).to(self.device)
        t1d = t1d.float().unsqueeze(0).to(self.device)
        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d*same_chain.float()[:,None]

        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, mask_in=mask_t.reshape(-1,L,27))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,effL,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,effL,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, effL, 30)

        T = xyz_t.shape[1]
        xyz_prev = xyz_t[:,0]
        mask_prev = mask_t[:,0]

        self.model.eval()
        for i_trial in range(NMODEL):
            start_time = time.time()
            self.run_prediction(
                native,
                msa_orig, ins_orig, t1d, t2d, xyz_t[:,:,:,1], alpha_t, mask_t_2d, xyz_prev, mask_prev, same_chain, idx_pdb,
                symmids, symmsub, symmRs, symmmeta,  "%s_%02d"%(out_prefix, i_trial))
            torch.cuda.empty_cache()

    def run_prediction(
        self, 
        native,
        msa_orig, ins_orig, t1d, t2d, xyz_t, alpha_t, mask_t, 
        xyz_prev, mask_prev, same_chain, idx_pdb, 
        symmids, symmsub, symmRs, symmmeta, out_prefix
    ):
        with torch.no_grad():
            if msa_orig.shape[0] > 10000:
                msa, ins = MSABlockDeletion(msa_orig, ins_orig)
                msa = msa.long().to(self.device) # (N, L)
                ins = ins.long().to(self.device)
            else:
                msa = msa_orig.long().to(self.device) # (N, L)
                ins = ins_orig.long().to(self.device)

            N, L = msa.shape[:2]
            O = symmids.shape[0]
            Osub = symmsub.shape[0]
            Lasu = L//Osub
            print ("Nseqs/L/Osub =", N, L, Osub)

            B = 1
            #
            idx_pdb = idx_pdb.to(self.device)
            symmids = symmids.to(self.device)
            symmsub = symmsub.to(self.device)
            symmRs = symmRs.to(self.device)

            subsymms, _ = symmmeta
            for i in range(len(subsymms)):
                subsymms[i] = subsymms[i].to(self.device)

            #self.write_pdb(msa[0], xyz_prev[0,:,:3], prefix="%s_templ"%(out_prefix), chainlen=Lasu)

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
                #self.write_pdb(seq[0], xyz_prev[0], Bfacts=pred_lddt[0], prefix="%s_cycle_%02d"%(out_prefix, i_cycle), chainlen=Lasu)

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


        outdata = {}

        # RMS
        monomer_rms, complex_rms, best_xyzfull = calc_symm_rmsd(best_xyzfull, native, O)
        outdata['monomer_rms'] = monomer_rms.item()
        outdata['complex_rms'] = complex_rms.item()
        outdata['mean_plddt'] = best_lddt.mean().item()
        for i in range(O):
            outdata['pae_chain0_'+str(i)] = 0.5 * (best_pae[:,0:Lasu,i*Lasu:(i+1)*Lasu].mean() + best_pae[:,i*Lasu:(i+1)*Lasu,0:Lasu].mean()).item()

        with open("%s.json"%(out_prefix), "w") as outfile:
            json.dump(outdata, outfile, indent=4)

        self.write_pdb(seq_full[0], best_xyzfull[0], Bfacts=best_lddtfull[0], prefix="%s_init"%(out_prefix), chainlen=Lasu)

        #prob_s = [prob.permute(0,2,3,1).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        #np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16))

    def write_pdb(self, seq, atoms, Bfacts=None, prefix=None, chainlen=None):
        L = len(seq)
        if chainlen is None:
            chainlen = L

        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)

            chains="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567"
            for i,s in enumerate(seq):
                i_seq = (i%chainlen)+1
                chn = chains[i//chainlen]
                if (len(atoms.shape)==2):
                    if (not torch.any(torch.isnan(atoms[i]))):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr%100000, " CA ", util.num2aa[s], 
                                chn, i_seq, atoms[i,0], atoms[i,1], atoms[i,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1

                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        if (not torch.any(torch.isnan(atoms[i,j]))):
                            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                    "ATOM", ctr%100000, atm_j, util.num2aa[s], 
                                    chn, i_seq, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                    1.0, Bfacts[i] ) )
                            ctr += 1                
                else:
                    atms = util.aa2long[s]
                    for j,atm_j in enumerate(atms):
                        if (atm_j is not None):
                            if (not torch.any(torch.isnan(atoms[i,j]))):
                                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                    "ATOM", ctr%100000, atm_j, util.num2aa[s], 
                                    chn, i_seq, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                    1.0, Bfacts[i] ) )
                                ctr += 1

def get_args():
    #DB="/home/robetta/rosetta_server_beta/external/databases/trRosetta/pdb100_2021Mar03/pdb100_2021Mar03"
    DB = "/projects/ml/TrRosetta/pdb100_2022Apr19/pdb100_2022Apr19"
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-pdb", required=True)
    parser.add_argument("-symm", required=False, default="C1")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    pred = Predictor()
    pred.predict(args.pdb, symm=args.symm)
