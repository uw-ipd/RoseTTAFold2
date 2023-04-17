import sys, os
from contextlib import ExitStack, nullcontext
import time
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex,
    Dataset, DatasetComplex, DistilledDataset, DistributedWeightedSampler
)
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, xyz_to_bbtor
from RoseTTAFoldModel  import RoseTTAFoldModule
from loss import *
from util import *
from util_module import XYZConverter
from scheduler import get_linear_schedule_with_warmup, get_stepwise_decay_schedule_with_warmup

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(5924)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## To reproduce errors
#import random
np.random.seed(6636)
#random.seed(0)

USE_AMP = False
torch.set_num_threads(4)

N_PRINT_TRAIN = 1
#BATCH_SIZE = 1 * torch.cuda.device_count()

# num structs per epoch
# must be divisible by #GPUs
N_EXAMPLE_PER_EPOCH = 25600

LOAD_PARAM = {'shuffle': False,
              'num_workers': 3,
              'pin_memory': True}

from arguments import get_args
args, model_param, loader_param, loss_param = get_args()


#define dataset & data loader
(
    pdb_items, fb_items, compl_items, neg_items, 
    valid_pdb, valid_homo, valid_compl, valid_neg, homo
) = get_train_valid_set(loader_param)

pdb_IDs, pdb_weights, pdb_dict = pdb_items
fb_IDs, fb_weights, fb_dict = fb_items
compl_IDs, compl_weights, compl_dict = compl_items
neg_IDs, neg_weights, neg_dict = neg_items

print ('Loaded (training)',
    len(pdb_IDs),'monomers/homomers,',
    len(fb_IDs),'distilled monomers,',
    len(compl_IDs),'heteromers,',
    len(neg_IDs),'negative heteromers,',
    len(homo),'Homomers',
)

for i,pdb_i in pdb_dict.items():
    #for j in range(len(pdb_i)):
    j = 0
    #print ('pdb',i,'.',j)
    loader_pdb(
        pdb_i[j][0],
        loader_param, 
        homo,
        unclamp=True, 
        pick_top=False, 
        p_homo_cut=1.0)


