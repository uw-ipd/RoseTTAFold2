import sys, os
from contextlib import ExitStack, nullcontext
import time
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
from data_loader import get_train_valid_set, loader_pdb, loader_fb, loader_complex, Dataset, DatasetComplex, DistilledDataset, DistributedWeightedSampler
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d
from RoseTTAFoldModel  import RoseTTAFoldModule
from loss import *
from util import *
from util_module import XYZConverter
from scheduler import get_stepwise_decay_schedule_with_warmup
from symmetry import symm_subunit_matrix, find_symm_subs, get_symm_map

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#torch.autograd.set_detect_anomaly(True)

USE_AMP = False

N_PRINT_TRAIN = 8
#BATCH_SIZE = 1 * torch.cuda.device_count()

# num structs per epoch
# must be divisible by #GPUs
N_EXAMPLE_PER_EPOCH = 3*3200


LOAD_PARAM = {'shuffle': False,
              'num_workers': 4,
              'pin_memory': True}
LOAD_PARAM2 = {'shuffle': False,
              'num_workers': 3,
              'pin_memory': True}

def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)

class Trainer():
    def __init__(self, model_name='BFF',
                 n_epoch=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, maxcycle=4):
        self.model_name = model_name #"BFF"
        #self.model_name = "%s_%d_%d_%d_%d"%(model_name, model_param['n_module'], 
        #                                    model_param['n_module_str'],
        #                                    model_param['d_msa'],
        #                                    model_param['d_pair'])
        #
        self.n_epoch = n_epoch
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        #
        self.model_param = model_param
        self.loader_param = loader_param
        self.valid_param = deepcopy(loader_param)
        self.valid_param['MINTPLT'] = 1
        self.valid_param['SEQID'] = 150.0
        self.loss_param = loss_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size

        # for all-atom str loss
        self.l2a = long2alt
        self.aamask = allatom_mask
        self.num_bonds = num_bonds
        self.ljlk_parameters = ljlk_parameters
        self.lj_correction_parameters = lj_correction_parameters
        self.hbtypes = hbtypes
        self.hbbaseatoms = hbbaseatoms
        self.hbpolys = hbpolys
        
        # from xyz to get xxxx or from xxxx to xyz
        self.xyz_converter = XYZConverter()

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        self.maxcycle = maxcycle

        print (model_param, loader_param, loss_param)
        
    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp, logit_pae, p_bind,
                  pred, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, unclamp=False, negative=False, pred_prev_s=None,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0, w_con=1.0, w_pae=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0, w_bind=0.0,
                  lj_lin=0.75, use_H=False, eps=1e-6, clashcut=0.0):
        B, L = true.shape[:2]
        seq = label_aa_s[:,0].clone()

        assert (B==1) # fd - code assumes a batch size of 1

        loss_s = list()
        tot_loss = 0.0

        # col 0~3: c6d loss (distogram, orientogram prediction)
        #fd for negatives, no loss on cross-chain
        if (negative):
            loss = calc_c6d_loss(logit_s, label_s, mask_2d*same_chain)
        else:
            loss = calc_c6d_loss(logit_s, label_s, mask_2d)
        tot_loss += w_dist * loss.sum()
        loss_s.append(loss.detach())

        # col 4: masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_s.append(loss[None].detach())

        # col 5: binder loss
        loss = torch.tensor(0.0,device=p_bind.device)
        if (torch.sum(same_chain==0) > 0):
            bce = torch.nn.BCELoss()
            target = torch.tensor([1.0],device=p_bind.device)
            if (negative):
                target = torch.tensor([0.0],device=p_bind.device)
            loss = bce(p_bind,target)
            tot_loss += w_bind * loss
            #print ( negative.detach().cpu().numpy(), p_bind.detach().cpu().numpy(), loss.detach().cpu().numpy() )
        else:
            # avoid unused parameter error
            tot_loss += 0.0 * p_bind.sum()
        loss_s.append(loss[None].detach())

        # update atom mask for structural loss calculation
        # calc lj for ground-truth --> ignore SC conformation from ground-truth if it makes clashes (lj > clashcut)
        xs_mask = self.aamask[seq] # (B, L, 27)
        xs_mask[0,:,14:]=False # ignore hydrogens
        xs_mask *= mask_crds # mask missing atoms & residues as well
        lj_nat = calc_lj(seq[0], true[0,..., :3], self.aamask, same_chain[0], 
                         self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
                         lj_lin=lj_lin, use_H=False, negative=negative, reswise=True, atom_mask=xs_mask[0])
        mask_clash = (lj_nat < clashcut) * mask_BB[0] # if False, the residue has clash (L)
        xs_mask[:,:,5:] *= mask_clash.view(1,L,1) # ignore clashed side-chains
        
        # col 5: experimentally resolved prediction loss
        loss = nn.BCEWithLogitsLoss()(logit_exp, mask_BB.float())
        tot_loss += w_exp*loss
        loss_s.append(loss[None].detach())

        # AllAtom loss
        # get ground-truth torsion angles
        true_tors, true_tors_alt, tors_mask, tors_planar = self.xyz_converter.get_torsions(true, seq, mask_in=xs_mask)
        # masking missing residues as well
        tors_mask *= mask_BB[...,None] # (B, L, 10)

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true)
        
        natRs_all, _n0 = self.xyz_converter.compute_all_atom(seq, true[...,:3,:], true_tors)
        natRs_all_alt, _n1 = self.xyz_converter.compute_all_atom(seq, true_alt[...,:3,:], true_tors_alt)
        predRs_all, pred_all = self.xyz_converter.compute_all_atom(seq, pred[-1], pred_tors[-1]) 

        #  - resolve symmetry
        natRs_all_symm, nat_symm = resolve_symmetry(pred_all[0], natRs_all[0], true[0], natRs_all_alt[0], true_alt[0], xs_mask[0])
        frame_mask = torch.cat( [mask_BB[0][:,None], tors_mask[0,:,:8]], dim=-1 ) # only first 8 torsions have unique frames
        
        # Structural loss
        # 1. Backbone FAPE
        if unclamp: 
            tot_str, str_loss, pae_loss  = calc_str_loss(pred, true, logit_pae,
                                                         mask_2d, same_chain, negative=negative,
                                                         A=10.0, d_clamp=None)
        else:
            tot_str, str_loss, pae_loss  = calc_str_loss(pred, true, logit_pae,
                                                         mask_2d, same_chain, negative=negative,
                                                         A=10.0, d_clamp=10.0)
        tot_loss += (1.0-w_all)*w_str*tot_str
        #loss_s.append(str_loss)
        
        tot_loss += w_pae * pae_loss
        loss_s.append(pae_loss[None].detach())
            
        # allatom fape and torsion angle loss
        if negative: # inter-chain fapes should be ignored for negative cases
            L1 = same_chain[0,0,:].sum()
            frame_maskA = frame_mask.clone()
            frame_maskA[L1:] = False
            xs_maskA = xs_mask.clone()
            xs_maskA[0, L1:] = False
            l_fape_A = compute_FAPE(
                predRs_all[0,frame_maskA][...,:3,:3], 
                predRs_all[0,frame_maskA][...,:3,3], 
                pred_all[xs_maskA][...,:3], 
                natRs_all_symm[frame_maskA][...,:3,:3], 
                natRs_all_symm[frame_maskA][...,:3,3], 
                nat_symm[xs_maskA[0]][...,:3],
                eps=1e-4)
            frame_maskB = frame_mask.clone()
            frame_maskB[:L1] = False
            xs_maskB = xs_mask.clone()
            xs_maskB[0,:L1] = False
            l_fape_B = compute_FAPE(
                predRs_all[0,frame_maskB][...,:3,:3], 
                predRs_all[0,frame_maskB][...,:3,3], 
                pred_all[xs_maskB][...,:3], 
                natRs_all_symm[frame_maskB][...,:3,:3], 
                natRs_all_symm[frame_maskB][...,:3,3], 
                nat_symm[xs_maskB[0]][...,:3],
                eps=1e-4)
            fracA = float(L1)/len(same_chain[0,0])
            l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B
        else:
            l_fape = compute_FAPE(
                predRs_all[0,frame_mask][...,:3,:3], 
                predRs_all[0,frame_mask][...,:3,3], 
                pred_all[xs_mask][...,:3], 
                natRs_all_symm[frame_mask][...,:3,:3], 
                natRs_all_symm[frame_mask][...,:3,3], 
                nat_symm[xs_mask[0]][...,:3],
                eps=1e-4)
        l_tors = torsionAngleLoss(
            pred_tors,
            true_tors,
            true_tors_alt,
            tors_mask,
            tors_planar,
            eps = 1e-10)
        tot_loss += w_all*w_str*(l_fape+l_tors)
        loss_s.append(l_fape[None].detach())
        loss_s.append(l_tors[None].detach())

        # CA-LDDT
        ca_lddt = calc_lddt(pred[:,:,:,1].detach(), true[:,:,1], mask_BB, mask_2d, same_chain, negative=negative)
        loss_s.append(ca_lddt.detach())
        
        # allatom lddt loss
        lddt_loss, true_lddt = calc_allatom_lddt_w_loss(pred_all[0,...,:14,:3].detach(), nat_symm[...,:14,:3], xs_mask[0,...,:14],
                                                        pred_lddt, idx[0], same_chain[0], negative=negative)
        loss_s.append(true_lddt[None].detach())
        tot_loss += w_lddt*lddt_loss
        loss_s.append(lddt_loss.detach()[None])
        
        # bond geometry
        blen_loss, bang_loss = calc_BB_bond_geom(pred[-1,:,:], idx)
        if w_blen > 0.0:
            tot_loss += w_blen*blen_loss
        if w_bang > 0.0:
            tot_loss += w_bang*bang_loss

        # lj potential
        lj_loss = calc_lj(
            seq[0], pred_all[0,...,:3], 
            self.aamask, same_chain[0], 
            self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds, #negative=negative,
            lj_lin=lj_lin, use_H=use_H)
        if w_lj > 0.0:
            tot_loss += w_lj*lj_loss

        # hbond [use all atoms not just those in native]
        hb_loss = calc_hb(
            seq[0], pred_all[0,...,:3], 
            self.aamask, self.hbtypes, self.hbbaseatoms, self.hbpolys)
        if w_hb > 0.0:
            tot_loss += w_hb*hb_loss

        loss_s.append(torch.stack((blen_loss, bang_loss, lj_loss, hb_loss)).detach())

        #if pred_prev_s != None:
        #    lddt_s = list()
        #    for pred_prev in pred_prev_s:
        #        prev_lddt = calc_allatom_lddt(pred_prev[0,:,:14,:3], nat_symm[:,:14,:3], xs_mask[0,:,:14],
        #                                      idx[0], same_chain[0], negative=negative)
        #        lddt_s.append(prev_lddt.detach())
        #    lddt_s.append(true_lddt.detach())
        #    lddt_s = torch.stack(lddt_s) 
        #    return tot_loss, lddt_s, torch.cat(loss_s, dim=0)
        #else:
        #    return tot_loss, true_lddt.detach(), torch.cat(loss_s, dim=0)

        # fd - for symmetry, ignore this for now (need to symmetrically resolve pred_prev_s)
        return tot_loss, true_lddt.detach(), torch.cat(loss_s, dim=0)

    def calc_acc(self, prob, dist, idx_pdb, mask_2d):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        mask *= mask_2d
        #
        cnt_ref = dist < 20
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:20,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        tmp_pred = torch.stack(tmp_pred, dim=0)
        tmp_pred = tmp_pred.float()*mask
        #
        condition = torch.logical_and(tmp_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (tmp_pred == torch.ones_like(tmp_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        
        return torch.stack([prec, recall, F1]), cnt_pred, cnt_ref

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=True):
        chk_fn = "models/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            #print ('no model found', model_name)
            return -1, best_valid_loss
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        #new_chk = {}
        #for param in model.module.model.state_dict():
        #    if param not in checkpoint['model_state_dict']:
        #        print ('missing',param)
        #        rename_model=True
        #    elif (checkpoint['model_state_dict'][param].shape == model.module.model.state_dict()[param].shape):
        #        new_chk[param] = checkpoint['model_state_dict'][param]
        #    else:
        #        print (
        #            'wrong size',param,
        #            checkpoint['model_state_dict'][param].shape,
        #             model.module.model.state_dict()[param].shape )

        #model.module.model.load_state_dict(checkpoint['final_state_dict'], strict=False)
        model.module.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.module.shadow.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if resume_train and (not rename_model):
            loaded_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists("models"):
            os.mkdir("models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join("models", name)
    
    # main entry function of training
    # 1) make sure ddp env vars set
    # 2) figure out if we launched using slurm or interactively
    #   - if slurm, assume 1 job launched per GPU
    #   - if interactive, launch one job for each GPU on node
    def run_model_training(self, world_size):
        if ('MASTER_ADDR' not in os.environ):
            os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.port

        if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def train_model(self, rank, world_size):
        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        pdb_items, fb_items, compl_items, neg_items, valid_pdb, valid_homo, valid_compl, valid_neg, homo = get_train_valid_set(self.loader_param)
        pdb_IDs, pdb_weights, pdb_dict = pdb_items
        fb_IDs, fb_weights, fb_dict = fb_items
        compl_IDs, compl_weights, compl_dict = compl_items
        neg_IDs, neg_weights, neg_dict = neg_items
        
        self.n_train = N_EXAMPLE_PER_EPOCH
        self.n_valid_pdb = len(valid_pdb.keys())
        self.n_valid_pdb = (self.n_valid_pdb // world_size)*world_size
        self.n_valid_homo = len(valid_homo.keys())
        self.n_valid_homo = (self.n_valid_homo // world_size)*world_size
        self.n_valid_compl = len(valid_compl.keys())
        self.n_valid_compl = (self.n_valid_compl // world_size)*world_size
        self.n_valid_neg = len(valid_neg.keys())
        self.n_valid_neg = (self.n_valid_neg // world_size)*world_size
        
        train_set = DistilledDataset(pdb_IDs, loader_pdb, pdb_dict,
                                     compl_IDs, loader_complex, compl_dict,
                                     neg_IDs, loader_complex, neg_dict,
                                     fb_IDs, loader_fb, fb_dict,
                                     homo, self.loader_param, p_homo_cut=2.0)
        valid_pdb_set = Dataset(list(valid_pdb.keys())[:self.n_valid_pdb],
                                loader_pdb, valid_pdb,
                                self.valid_param, homo, p_homo_cut=-1.0)
        valid_homo_set = Dataset(list(valid_homo.keys())[:self.n_valid_homo],
                                loader_pdb, valid_homo,
                                self.valid_param, homo, p_homo_cut=2.0)
        valid_compl_set = DatasetComplex(list(valid_compl.keys())[:self.n_valid_compl],
                                         loader_complex, valid_compl,
                                         self.valid_param, negative=False)
        valid_neg_set = DatasetComplex(list(valid_neg.keys())[:self.n_valid_neg],
                                        loader_complex, valid_neg,
                                        self.valid_param, negative=True)
        #
        train_sampler = DistributedWeightedSampler(train_set, pdb_weights, compl_weights, neg_weights, fb_weights, 
                                                   num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
                                                   num_replicas=world_size, rank=rank, fraction_fb=0.0, fraction_compl=0.0)
        valid_pdb_sampler = data.distributed.DistributedSampler(valid_pdb_set, num_replicas=world_size, rank=rank)
        valid_homo_sampler = data.distributed.DistributedSampler(valid_homo_set, num_replicas=world_size, rank=rank)
        valid_compl_sampler = data.distributed.DistributedSampler(valid_compl_set, num_replicas=world_size, rank=rank)
        valid_neg_sampler = data.distributed.DistributedSampler(valid_neg_set, num_replicas=world_size, rank=rank)
       
        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
        valid_pdb_loader = data.DataLoader(valid_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
        valid_homo_loader = data.DataLoader(valid_homo_set, sampler=valid_homo_sampler, **LOAD_PARAM2)
        valid_compl_loader = data.DataLoader(valid_compl_set, sampler=valid_compl_sampler, **LOAD_PARAM)
        valid_neg_loader = data.DataLoader(valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM)

        # move some global data to cuda device
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        self.xyz_converter = self.xyz_converter.to(gpu)
        self.num_bonds = self.num_bonds.to(gpu)
        self.ljlk_parameters = self.ljlk_parameters.to(gpu)
        self.lj_correction_parameters = self.lj_correction_parameters.to(gpu)
        self.hbtypes = self.hbtypes.to(gpu)
        self.hbbaseatoms = self.hbbaseatoms.to(gpu)
        self.hbpolys = self.hbpolys.to(gpu)

        # define model
        model = EMA(RoseTTAFoldModule(**self.model_param).to(gpu), 0.99)

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        #ddp_model._set_static_graph() # required to use gradient checkpointing w/ shared parameters
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))
        
        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        #optimizer = torch.optim.Adam(opt_params, lr=self.init_lr)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        #scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 15000, 0.95)
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 10000, 0.95)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
       
        # load model
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                        self.model_name, gpu, resume_train=False)

        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return
        
        valid_pdb_sampler.set_epoch(0)
        #valid_homo_sampler.set_epoch(0)
        #valid_compl_sampler.set_epoch(0)
        #valid_neg_sampler.set_epoch(0)
        #valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, loaded_epoch)
        _, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, loaded_epoch, header="Homo")
        #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, loaded_epoch)
        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            valid_pdb_sampler.set_epoch(epoch)
            valid_homo_sampler.set_epoch(epoch)
            valid_compl_sampler.set_epoch(epoch)
            valid_neg_sampler.set_epoch(epoch)
          
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)
            valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, epoch)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, epoch, header="Homo")
            #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch)

            if rank == 0: # save model
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save({'epoch': epoch,
                                #'model_state_dict': ddp_model.state_dict(),
                                'model_state_dict': ddp_model.module.shadow.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'best_loss': best_valid_loss,
                                'train_loss': train_loss,
                                'train_acc': train_acc,
                                'valid_loss': valid_loss,
                                'valid_acc': valid_acc},
                                self.checkpoint_fn(self.model_name, 'best'))
            
            
                torch.save({'epoch': epoch,
                            #'model_state_dict': ddp_model.state_dict(),
                            'model_state_dict': ddp_model.module.shadow.state_dict(),
                            'final_state_dict': ddp_model.module.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': valid_loss,
                            'valid_acc': valid_acc,
                            'best_loss': best_valid_loss},
                            self.checkpoint_fn(self.model_name, 'last'))
                
        dist.destroy_process_group()
    
    def _prepare_input(self, inputs, gpu):
        seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, symmgp = inputs
        
        # transfer inputs to device
        B, _, N, L = msa.shape

        idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
        true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
        mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
        same_chain = same_chain.to(gpu, non_blocking=True)

        xyz_t = xyz_t.to(gpu, non_blocking=True)
        t1d = t1d.to(gpu, non_blocking=True)
        mask_t = mask_t.to(gpu, non_blocking=True)
        
        xyz_prev = xyz_prev.to(gpu, non_blocking=True)
        mask_prev = mask_prev.to(gpu, non_blocking=True)

        seq = seq.to(gpu, non_blocking=True)
        msa = msa.to(gpu, non_blocking=True)
        msa_masked = msa_masked.to(gpu, non_blocking=True)
        msa_full = msa_full.to(gpu, non_blocking=True)
        mask_msa = mask_msa.to(gpu, non_blocking=True)

        assert (len(symmgp)==1)
        symmgp = symmgp[0]

        # symmetry - reprocess (many) inputs
        if (symmgp != 'C1'):
            Lasu = L//2 # msa contains intra/inter block
            symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix(symmgp)
            symmids = symmids.to(gpu, non_blocking=True)
            symmRs = symmRs.to(gpu, non_blocking=True)
            symmoffset = symmoffset.to(gpu, non_blocking=True)
            O = symmids.shape[0]
            xyz_prev = xyz_prev + symmoffset*Lasu**(1/3)

            # find contacting subunits
            xyz_prev, symmsub = find_symm_subs(xyz_prev[:,:Lasu], symmRs, symmmeta)
            symmsub = symmsub.to(gpu, non_blocking=True)
            Osub = symmsub.shape[0]
            mask_prev = mask_prev[:,:L].repeat(1,Osub,1)

            # symmetrize msa
            seq = torch.cat([seq[:,:,:Lasu],*[seq[:,:,Lasu:]]*(Osub-1)], dim=2)
            msa = torch.cat([msa[:,:,:,:Lasu],*[msa[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            msa_masked = torch.cat([msa_masked[:,:,:,:Lasu],*[msa_masked[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            msa_full = torch.cat([msa_full[:,:,:,:Lasu],*[msa_full[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            mask_msa = torch.cat([mask_msa[:,:,:,:Lasu],*[mask_msa[:,:,:,Lasu:]]*(Osub-1)], dim=3)

            # symmetrize templates
            #print (xyz_t.shape)
            xyz_t = xyz_t[:,:,:Lasu].repeat(1,1,Osub,1,1)
            mask_t = mask_t[:,:,:Lasu].repeat(1,1,Osub,1)
            t1d = t1d[:,:,:Lasu].repeat(1,1,Osub,1)

            # index, same chain
            idx_pdb = torch.arange(Osub*Lasu, device=gpu)[None,:]
            same_chain = torch.zeros((1,Osub*Lasu,Osub*Lasu), device=gpu).long()
            for o_i in range(Osub):
                i = symmsub[o_i]
                same_chain[:,o_i*Lasu:(i+1)*Lasu,o_i*Lasu:(o_i+1)*Lasu] = 1
                idx_pdb[:,o_i*Lasu:(i+1)*Lasu] += 100*o_i

            #print ('sym',symmgp,Osub,seq.shape,xyz_t.shape)
        else:
            Lasu = L
            Osub = 1
            symmids = None
            symmsub = None
            symmRs = None
            symmmeta = None
            #print ('asym',symmgp,Osub,seq.shape,xyz_t.shape)

        # processing template features
        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,Lasu*Osub)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,Lasu*Osub,27,3), seq_tmp, mask_in=mask_t.reshape(-1,Lasu*Osub,27))
        alpha = alpha.reshape(B,-1,Lasu*Osub,10,2)
        alpha_mask = alpha_mask.reshape(B,-1,Lasu*Osub,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, Lasu*Osub, 30)

        network_input = {}
        network_input['msa_latent'] = msa_masked
        network_input['msa_full'] = msa_full
        network_input['seq'] = seq
        network_input['idx'] = idx_pdb
        network_input['t1d'] = t1d
        network_input['t2d'] = t2d
        network_input['xyz_t'] = xyz_t[:,:,:,1]
        network_input['alpha_t'] = alpha_t
        network_input['mask_t'] = mask_t_2d
        network_input['same_chain'] = same_chain

        network_input['symmids'] = symmids
        network_input['symmsub'] = symmsub
        network_input['symmRs'] = symmRs
        network_input['symmmeta'] = symmmeta

        mask_recycle = mask_prev[:,:,:3].bool().all(dim=-1)
        mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
        mask_recycle = same_chain.float()*mask_recycle.float()
        return network_input, xyz_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu
    
    def _get_model_input(self, network_input, output_i, i_cycle, return_raw=False, use_checkpoint=False):
        input_i = {}
        for key in network_input:
            if key in ['msa_latent', 'msa_full', 'seq']:
                input_i[key] = network_input[key][:,i_cycle]
            else:
                input_i[key] = network_input[key]
        msa_prev, pair_prev, state_prev, xyz_prev, alpha, mask_recycle = output_i
        input_i['msa_prev'] = msa_prev
        input_i['pair_prev'] = pair_prev
        input_i['state_prev'] = state_prev
        input_i['xyz'] = xyz_prev
        input_i['mask_recycle'] = mask_recycle
        input_i['return_raw'] = return_raw
        input_i['use_checkpoint'] = use_checkpoint
        return input_i

    def _get_loss_and_misc(self, output_i, true_crds, mask_crds, same_chain,
                           msa, mask_msa, idx_pdb, unclamp, negative, symmRs, Lasu,
                           pred_prev_s=None, return_cnt=False):
        logit_s, logit_aa_s, logit_exp, logit_pae, p_bind, pred_crds, alphas, symmsubs, pred_lddts, _, _, _ = output_i
            
        if (symmRs is not None):
            #print ('a', pred_crds.shape, true_crds.shape, mask_crds.shape)
            ###
            # resolve symmetry
            ###
            true_crds = true_crds[:,0]
            mask_crds = mask_crds[:,0]
            mapT2P = resolve_symmetry_predictions(pred_crds, true_crds, mask_crds, Lasu) # (Nlayer, Ltrue)

            # update all derived data to only include subunits mapping to native
            logit_s_new = []
            for li in logit_s:
                li=torch.gather(li,2,mapT2P[-1][None,None,:,None].repeat(1,li.shape[1],1,li.shape[-1]))
                li=torch.gather(li,3,mapT2P[-1][None,None,None,:].repeat(1,li.shape[1],li.shape[2],1))
                logit_s_new.append(li)
            logit_s = tuple(logit_s_new)

            logit_aa_s = logit_aa_s.view(1,21,msa.shape[-2],msa.shape[-1])
            logit_aa_s = torch.gather(logit_aa_s,3,mapT2P[-1][None,None,None,:].repeat(1,21,logit_aa_s.shape[-2],1))
            logit_aa_s = logit_aa_s.view(1,21,-1)

            msa = torch.gather(msa,2,mapT2P[-1][None,None,:].repeat(1,msa.shape[-2],1))
            mask_msa = torch.gather(mask_msa,2,mapT2P[-1][None,None,:].repeat(1,mask_msa.shape[-2],1))
            logit_exp = torch.gather(logit_exp,1,mapT2P[-1][None,:])

            logit_pae=torch.gather(logit_pae,2,mapT2P[-1][None,None,:,None].repeat(1,logit_pae.shape[1],1,logit_pae.shape[-1]))
            logit_pae=torch.gather(logit_pae,3,mapT2P[-1][None,None,None,:].repeat(1,logit_pae.shape[1],logit_pae.shape[2],1))

            pred_crds = torch.gather(pred_crds,2,mapT2P[:,None,:,None,None].repeat(1,1,1,27,3))
            alphas = torch.gather(alphas,2,mapT2P[:,None,:,None,None].repeat(1,1,1,10,2))

            same_chain=torch.gather(same_chain,1,mapT2P[-1][None,:,None].repeat(1,1,same_chain.shape[-1]))
            same_chain=torch.gather(same_chain,2,mapT2P[-1][None,None,:].repeat(1,same_chain.shape[1],1))

            pred_lddts = torch.gather(pred_lddts,2,mapT2P[-1][None,None,:].repeat(1,pred_lddts.shape[-2],1))
            idx_pdb = torch.gather(idx_pdb,1,mapT2P[-1][None,:])
        else:
            # find closest homo-oligomer pairs
            #print ('b', pred_crds.shape, true_crds.shape, mask_crds.shape)
            true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
        #print ('c', pred_crds.shape, true_crds.shape, mask_crds.shape)

        # processing labels for distogram orientograms
        mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
        mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
        c6d = xyz_to_c6d(true_crds)
        c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

        prob = self.active_fn(logit_s[0]) # distogram
        acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

        loss, lddt, loss_s = self.calc_loss(logit_s, c6d,
                    logit_aa_s, msa, mask_msa, logit_exp, logit_pae, p_bind,
                    pred_crds, alphas, true_crds, mask_crds,
                    mask_BB, mask_2d, same_chain,
                    pred_lddts, idx_pdb, unclamp=unclamp, negative=negative,
                    pred_prev_s=pred_prev_s,
                    **self.loss_param)
        
        if return_cnt:
            return loss, lddt, loss_s, acc_s, cnt_pred, cnt_ref
        else:
            return loss, lddt, loss_s, acc_s

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        for inputs in train_loader:
            network_input, xyz_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu = self._prepare_input(inputs, gpu)

            counter += 1

            N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

            output_i = (None, None, None, xyz_prev, None, mask_recycle)
            for i_cycle in range(N_cycle):
                with ExitStack() as stack:
                    if i_cycle < N_cycle -1:
                        stack.enter_context(torch.no_grad())
                        stack.enter_context(ddp_model.no_sync())
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw=True
                        use_checkpoint=False
                    else:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw=False
                        use_checkpoint=True
                    
                    input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw, use_checkpoint=use_checkpoint)

                    output_i = ddp_model(**input_i)
                    
                    if i_cycle < N_cycle - 1:
                        continue

                    loss, _, loss_s, acc_s = self._get_loss_and_misc(output_i,
                                               true_crds, mask_crds, network_input['same_chain'],
                                               msa[:,i_cycle], mask_msa[:,i_cycle],
                                               network_input['idx'],
                                               unclamp, negative, symmRs, Lasu)

            loss = loss / self.ACCUM_STEP
            scaler.scale(loss).backward()
            if counter%self.ACCUM_STEP == 0:  
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()
                ddp_model.module.update() # apply EMA
            
            ## check parameters with no grad
            #if rank == 0:
            #    for n, p in ddp_model.named_parameters():
            #        if p.grad is None and p.requires_grad is True:
            #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model
            

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss == None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += loss_s.detach()
            local_acc += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss == None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach()
            train_acc += acc_s.detach()
            
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time = time.time() - start_time
                    local_tot /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"%(\
                            epoch, self.n_epoch, counter*self.batch_size*world_size, self.n_train, train_time, local_tot, \
                            " ".join(["%8.4f"%l for l in local_loss]),\
                            local_acc[0], local_acc[1], local_acc[2], max_mem))
                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None 
                    local_acc = None 
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # write total train loss
        train_tot /= float(counter * world_size)
        train_loss /= float(counter * world_size)
        train_acc  /= float(counter * world_size)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_tot = train_tot.cpu().detach()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()
        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()
            
        return train_tot, train_loss, train_acc
    
    def valid_pdb_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch, header='PDB'):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        counter = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_loader:
                network_input, xyz_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                output_i = (None, None, None, xyz_prev, None, mask_recycle)
                pred_prev_s = list()
                for i_cycle in range(N_cycle):
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint=False
                        if i_cycle < N_cycle -1:
                            return_raw=True
                        else:
                            return_raw=False
                        
                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)

                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            predTs = output_i[3]
                            pred_tors = output_i[4]
                            _, pred_all = self.xyz_converter.compute_all_atom(msa[:,i_cycle,0], predTs, pred_tors)
                            pred_prev_s.append(pred_all.detach())
                            continue
                        
                        loss, lddt_s, loss_s, acc_s = self._get_loss_and_misc(output_i,
                                                    true_crds, mask_crds, network_input['same_chain'],
                                                    msa[:,i_cycle], mask_msa[:,i_cycle],
                                                    network_input['idx'],
                                                    unclamp, negative, symmRs, Lasu,
                                                    pred_prev_s)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()

        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        valid_lddt /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, counter*world_size, counter*world_size, 
                    train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc
    
    def valid_ppi_cycle(self, ddp_model, valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch, verbose=False):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        valid_inter = None
        counter = 0
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_pos_loader:
                network_input, xyz_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu = self._prepare_input(inputs, gpu)
                
                counter += 1

                N_cycle = self.maxcycle # number of recycling

                output_i = (None, None, None, xyz_prev, None, mask_recycle)
                pred_prev_s = list()
                for i_cycle in range(N_cycle): 
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint=False
                        if i_cycle < N_cycle - 1:
                            return_raw=True
                        else:
                            return_raw=False
                        
                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle-1:
                            predTs = output_i[3]
                            pred_tors = output_i[4]
                            _, pred_all = self.xyz_converter.compute_all_atom(msa[:,i_cycle,0], predTs, pred_tors)
                            pred_prev_s.append(pred_all.detach())
                            continue

                        loss, lddt_s, loss_s, acc_s, cnt_pred, cnt_ref = self._get_loss_and_misc(output_i,
                                                                                true_crds, mask_crds, network_input['same_chain'],
                                                                                msa[:,i_cycle], mask_msa[:,i_cycle],
                                                                                network_input['idx'],
                                                                                unclamp, negative, symmRs, Lasu, pred_prev_s,
                                                                                return_cnt=True)
                
                        # inter-chain contact prob
                        cnt_pred = cnt_pred * (1-network_input['same_chain']).float()
                        cnt_ref = cnt_ref * (1-network_input['same_chain']).float()
                        max_prob = cnt_pred.max()
                        if max_prob > 0.5:
                            if (cnt_ref > 0).any():
                                TP += 1.0
                            else:
                                FP += 1.0
                        else:
                            if (cnt_ref > 0).any():
                                FN += 1.0
                            else:
                                TN += 1.0
                        inter_s = torch.tensor([TP, FP, TN, FN], device=cnt_pred.device).float()

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                    valid_inter = torch.zeros_like(inter_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()
                valid_inter += inter_s.detach()
            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        valid_lddt /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        
        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write("Hetero: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0

        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_neg_loader: 
                network_input, xyz_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle # number of recycling
                
                output_i = (None, None, None, xyz_prev, None, mask_recycle)
                pred_prev_s = list()
                for i_cycle in range(N_cycle): 
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        if i_cycle < N_cycle - 1:
                            return_raw=True
                        else:
                            return_raw=False

                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            predTs = output_i[3]
                            pred_tors = output_i[4]
                            _, pred_all = self.xyz_converter.compute_all_atom(msa[:,i_cycle,0], predTs, pred_tors)
                            pred_prev_s.append(pred_all.detach())
                            continue

                        loss, lddt_s, loss_s, acc_s, cnt_pred, cnt_ref = self._get_loss_and_misc(output_i,
                                                                                true_crds, mask_crds, network_input['same_chain'],
                                                                                msa[:,i_cycle], mask_msa[:,i_cycle],
                                                                                network_input['idx'],
                                                                                unclamp, negative, symmRs, Lasu, pred_prev_s,
                                                                                return_cnt=True)
                        # inter-chain contact prob
                        cnt_pred = cnt_pred * (1-network_input['same_chain']).float()
                        cnt_ref = cnt_ref * (1-network_input['same_chain']).float()
                        max_prob = cnt_pred.max()
                        if max_prob > 0.5:
                            if (cnt_ref > 0).any():
                                TP += 1.0
                            else:
                                FP += 1.0
                        else:
                            if (cnt_ref > 0).any():
                                FN += 1.0
                            else:
                                TN += 1.0
                        inter_s = torch.tensor([TP, FP, TN, FN], device=cnt_pred.device).float()

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_inter, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_inter = valid_inter.cpu().detach().numpy()
        
        if rank == 0:
            TP, FP, TN, FN = valid_inter 
            prec = TP/(TP+FP+1e-4)
            recall = TP/(TP+FN+1e-4)
            F1 = 2*TP/(2*TP+FP+FN+1e-4)
            
            train_time = time.time() - start_time
            sys.stdout.write("PPI: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2],\
                    prec, recall, F1))
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc

if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param = get_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mp.freeze_support()
    train = Trainer(model_name=args.model_name,
                    interactive=args.interactive,
                    n_epoch=args.num_epochs, lr=args.lr, l2_coeff=1.0e-4,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum,
                    maxcycle=args.maxcycle)
    train.run_model_training(torch.cuda.device_count())
