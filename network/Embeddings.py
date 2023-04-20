import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.utils.checkpoint as checkpoint
from util import get_Cb
from util_module import Dropout, create_custom_forward, rbf, init_lecun_normal
from Attention_module import Attention, FeedForwardLayer, AttentionWithBias
from Track_module import PairStr2Pair

# Module contains classes and functions to generate initial embeddings
class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        #self.emb_chain = nn.Embedding(2, d_model)
    
    def forward(self, idx):
        B, L = idx.shape[:2]

        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)

        seqsep = torch.full((B,L,L),100, device=idx.device)
        seqsep[0] = idx[0,None,:] - idx[0,:,None] # (B, L, L)

        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        #emb_c = self.emb_chain(same_chain.long())
        return emb #+ emb_c

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=22+22+2+2,
                 minpos=-32, maxpos=32, p_drop=0.1):
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(22, d_state)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos)

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, symmids=None):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        B, N, L = msa.shape[:3] # number of sequenes in MSA

        # msa embedding 
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        tmp = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA

        # pair embedding 
        left = self.emb_left(seq)[:,None] # (B, 1, L, d_pair)
        right = self.emb_right(seq)[:,:,None] # (B, L, 1, d_pair)
        pair = (left + right) # (B, L, L, d_pair)
        pair = pair + self.pos(idx) # add relative position

        # state embedding
        state = self.emb_state(seq) #.repeat(oligo,1,1)

        return msa, pair, state

class Extra_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=22+1+2, p_drop=0.1):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, oligo=1):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #N = msa.shape[1] # number of sequenes in MSA

        B,N,L = msa.shape[:3]

        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA

        return msa 

class TemplatePairStack(nn.Module):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, d_t1d=22, d_state=32, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        self.proj_t1d = nn.Linear(d_t1d, d_state)
        proc_s = [PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, d_state=d_state, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
        self.reset_parameter()
    
    def reset_parameter(self):
        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, templ, rbf_feat, t1d, use_checkpoint=False, p2p_crop=-1, symmids=None):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B*T, L, L, -1)
        t1d = t1d.reshape(B*T, L, -1)
        state = self.proj_t1d(t1d)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block]), templ, rbf_feat, state, p2p_crop) #, symmids)
            else:
                templ = self.block[i_block](templ, rbf_feat, state, p2p_crop) #, symmids)
        return self.norm(templ).reshape(B, T, L, L, -1)


class TemplateTorsionStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, d_rbf=64, n_head=4, d_hidden=16, p_drop=0.15):
        super(TemplateTorsionStack, self).__init__()
        self.n_block=n_block
        self.proj_pair = nn.Linear(d_templ+d_rbf, d_templ)
        proc_s = [AttentionWithBias(d_in=d_templ, d_bias=d_templ,
                                    n_head=n_head, d_hidden=d_hidden) for i in range(n_block)]
        self.row_attn = nn.ModuleList(proc_s)
        proc_s = [FeedForwardLayer(d_templ, 4, p_drop=p_drop) for i in range(n_block)]
        self.ff = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)

    def reset_parameter(self):
        self.proj_pair = init_lecun_normal(self.proj_pair)
        nn.init.zeros_(self.proj_pair.bias)

    def forward(self, tors, pair, rbf_feat, use_checkpoint=False):
        B, T, L = tors.shape[:3]
        tors = tors.reshape(B*T, L, -1)
        pair = pair.reshape(B*T, L, L, -1)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)
        
        for i_block in range(self.n_block):
            if use_checkpoint:
                tors = tors + checkpoint.checkpoint(create_custom_forward(self.row_attn[i_block]), tors, pair)
            else:
                tors = tors + self.row_attn[i_block](tors, pair)
            tors = tors + self.ff[i_block](tors)
        return self.norm(tors).reshape(B, T, L, -1)

class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   
    def __init__(self, d_t1d=21+1, d_t2d=43+1, d_tor=30, d_pair=128, d_state=32, 
                 n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super(Templ_emb, self).__init__()
        # process 2D features
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop)
        
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair, p_drop=p_drop)
        
        # process torsion angles
        self.proj_t1d = nn.Linear(d_t1d+d_tor, d_templ)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state, p_drop=p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

        nn.init.kaiming_normal_(self.proj_t1d.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_t1d.bias)
    
    def _get_templ_emb(self, t1d, t2d):
        B, T, L, _ = t1d.shape
        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 88)
        return self.emb(templ) # Template templures (B, T, L, L, d_templ)
        
    def _get_templ_rbf(self, xyz_t, mask_t):
        B, T, L = xyz_t.shape[:3]

        # process each template features
        xyz_t = xyz_t.reshape(B*T, L, 3)
        mask_t = mask_t.reshape(B*T, L, L)
        rbf_feat = rbf(torch.cdist(xyz_t, xyz_t)).to(xyz_t.dtype) * mask_t[...,None] # (B*T, L, L, d_rbf)
        return rbf_feat
    
    def forward(self, t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, use_checkpoint=False, p2p_crop=-1, symmids=None):
        # Input
        #   - t1d: 1D template info (B, T, L, 22)
        #   - t2d: 2D template info (B, T, L, L, 44)
        #   - alpha_t: torsion angle info (B, T, L, 30)
        #   - xyz_t: template CA coordinates (B, T, L, 3)
        #   - mask_t: is valid residue pair? (B, T, L, L)
        #   - pair: query pair features (B, L, L, d_pair)
        #   - state: query state features (B, L, d_state)
        B, T, L, _ = t1d.shape
        
        templ = self._get_templ_emb(t1d, t2d)
        rbf_feat = self._get_templ_rbf(xyz_t, mask_t)
        
        # process each template pair feature
        templ = self.templ_stack(
            templ, rbf_feat, t1d, use_checkpoint=use_checkpoint, p2p_crop=p2p_crop, symmids=symmids
        ).to(pair.dtype) # (B, T, L,L, d_templ)

        # Prepare 1D template torsion angle features
        t1d = torch.cat((t1d, alpha_t), dim=-1) # (B, T, L, 22+30)
        t1d = self.proj_t1d(t1d)
        
        # mixing query state features to template state features
        state = state.reshape(B*L, 1, -1) # (B*L, 1, d_state)
        t1d = t1d.permute(0,2,1,3).reshape(B*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn_tor), state, t1d, t1d)
            out = out.reshape(B, L, -1)
        else:
            out = self.attn_tor(state, t1d, t1d).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        state = state + out

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair, state

class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_rbf=64):
        super(Recycling, self).__init__()
        #self.emb_rbf = nn.Linear(d_rbf, d_pair)
        self.proj_dist = nn.Linear(d_rbf+d_state*2, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        #self.emb_rbf = init_lecun_normal(self.emb_rbf)
        #nn.init.zeros_(self.emb_rbf.bias)
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)

    def forward(self, seq, msa, pair, state, xyz, mask_recycle=None):
        B, L = msa.shape[:2]
        state = self.norm_state(state)
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        ## SYMM
        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)

        # recreate Cb given N,Ca,C
        Cb = get_Cb(xyz[:,:,:3])

        dist_CB = rbf(
            torch.cdist(Cb, Cb)
        ).reshape(B,L,L,-1)

        if mask_recycle != None:
            dist_CB = mask_recycle[...,None].float()*dist_CB

        dist_CB = torch.cat((dist_CB, left, right), dim=-1)
        dist = self.proj_dist(dist_CB)

        pair = pair + dist 

        return msa, pair, state

