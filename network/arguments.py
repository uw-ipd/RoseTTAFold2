import argparse
import data_loader
import os

TRUNK_PARAMS = ['n_extra_block', 'n_main_block', 'n_ref_block',\
                'd_msa', 'd_msa_full', 'd_pair', 'd_templ',\
                'n_head_msa', 'n_head_pair', 'n_head_templ', 'd_hidden', 'd_hidden_templ', 'p_drop']

base_SE3 = ['num_layers', 'num_channels', 'l0_in_features', 'l0_out_features', 'l1_in_features', 'l1_out_features', 'num_edge_features']
SE3_PARAMS = ['num_degrees', 'n_heads', 'div']
for se3 in base_SE3:
    for suffix in ['full', 'topk', 'SC']:
        SE3_PARAMS.append("%s_%s"%(se3, suffix))

def get_args():
    parser = argparse.ArgumentParser()

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument("-model_name", default="BFF",
            help="model name for saving")
    train_group.add_argument('-batch_size', type=int, default=1,
            help="Batch size [1]")
    train_group.add_argument('-lr', type=float, default=1.0e-3, 
            help="Learning rate [1.0e-3]")
    train_group.add_argument('-num_epochs', type=int, default=200,
            help="Number of epochs [200]")
    train_group.add_argument("-port", type=int, default=12319,
            help="PORT for ddp training, should be randomized [12319]")
    train_group.add_argument("-seed", type=int, default=0,
            help="seed for random number, should be randomized for different training run [0]")
    train_group.add_argument("-accum", type=int, default=1,
            help="Gradient accumulation when it's > 1 [1]")
    train_group.add_argument("-interactive", action="store_true", default=False,
            help="Use interactive node")

    # data-loading parameters
    data_group = parser.add_argument_group("data loading parameters")
    data_group.add_argument('-maxseq', type=int, default=1024,
            help="Maximum depth of subsampled MSA [1024]")
    data_group.add_argument('-maxlat', type=int, default=128,
            help="Maximum depth of subsampled MSA [128]")
    data_group.add_argument("-crop", type=int, default=256,
            help="Upper limit of crop size [256]")
    data_group.add_argument('-mintplt', type=int, default=0,
            help="Minimum number of templates to select [0]")
    data_group.add_argument('-maxtplt', type=int, default=4,
            help="maximum number of templates to select [4]")
    data_group.add_argument("-rescut", type=float, default=5.0,
            help="Resolution cutoff [5.0]")
    data_group.add_argument("-datcut", default="2020-Apr-30",
            help="PDB release date cutoff [2020-Apr-30]")
    data_group.add_argument('-plddtcut', type=float, default=70.0,
            help="pLDDT cutoff for distillation set [70.0]")
    data_group.add_argument('-seqid', type=float, default=150.0,
            help="maximum sequence identity cutoff for template selection [150.0]")
    data_group.add_argument('-maxcycle', type=int, default=4,
            help="maximum number of recycle [4]")

    # Trunk module properties
    trunk_group = parser.add_argument_group("Trunk module parameters")
    trunk_group.add_argument('-n_extra_block', type=int, default=4,
            help="Number of iteration blocks for extra sequences [4]")
    trunk_group.add_argument('-n_main_block', type=int, default=48,
            help="Number of iteration blocks for main sequences [48]")
    trunk_group.add_argument('-n_ref_block', type=int, default=4,
            help="Number of refinement layers [4]")
    trunk_group.add_argument('-d_msa', type=int, default=256,
            help="Number of MSA features [256]")
    trunk_group.add_argument('-d_msa_full', type=int, default=64,
            help="Number of MSA features [64]")
    trunk_group.add_argument('-d_pair', type=int, default=128,
            help="Number of pair features [128]")
    trunk_group.add_argument('-d_templ', type=int, default=64,
            help="Number of templ features [64]")
    trunk_group.add_argument('-n_head_msa', type=int, default=8,
            help="Number of attention heads for MSA2MSA [8]")
    trunk_group.add_argument('-n_head_pair', type=int, default=4,
            help="Number of attention heads for Pair2Pair [4]")
    trunk_group.add_argument('-n_head_templ', type=int, default=4,
            help="Number of attention heads for template [4]")
    trunk_group.add_argument("-d_hidden", type=int, default=32,
            help="Number of hidden features [32]")
    trunk_group.add_argument("-d_hidden_templ", type=int, default=32,
            help="Number of hidden features for templates [32]")
    trunk_group.add_argument("-p_drop", type=float, default=0.15,
            help="Dropout ratio [0.15]")

    # Structure module properties
    str_group = parser.add_argument_group("structure module parameters")
    str_group.add_argument('-num_degrees', type=int, default=2,
            help="Number of degrees for SE(3) network [2]")
    str_group.add_argument('-n_heads', type=int, default=4,
            help="Number of attention heads for SE3-Transformer [4]")
    str_group.add_argument("-div", type=int, default=4,
            help="Div parameter for SE3-Transformer [4]")

    str_group.add_argument('-num_layers_full', type=int, default=1,
            help="Number of equivariant layers in fully-connected structure module block [1]")
    str_group.add_argument('-num_channels_full', type=int, default=48,
            help="Number of channels in structure module block [48]")
    str_group.add_argument('-l0_in_features_full', type=int, default=32,
            help="Number of type 0 input features for full-connected graph [32]")
    str_group.add_argument('-l0_out_features_full', type=int, default=32,
            help="Number of type 0 output features for full-connected graph [32]")
    str_group.add_argument('-l1_in_features_full', type=int, default=2,
            help="Number of type 1 input features [2]")
    str_group.add_argument('-l1_out_features_full', type=int, default=2,
            help="Number of type 1 output features [2]")
    str_group.add_argument('-num_edge_features_full', type=int, default=32,
            help="Number of edge features for full-connected graph [32]")

    str_group.add_argument('-num_layers_topk', type=int, default=1,
            help="Number of equivariant layers in top-k structure module block [1]")
    str_group.add_argument('-num_channels_topk', type=int, default=128,
            help="Number of channels in structure module block [128]")
    str_group.add_argument('-l0_in_features_topk', type=int, default=64,
            help="Number of type 0 input features for top-k graph [64]")
    str_group.add_argument('-l0_out_features_topk', type=int, default=64,
            help="Number of type 0 output features for top-k graph [64]")
    str_group.add_argument('-l1_in_features_topk', type=int, default=2,
            help="Number of type 1 input features [2]")
    str_group.add_argument('-l1_out_features_topk', type=int, default=2,
            help="Number of type 1 output features [2]")
    str_group.add_argument('-num_edge_features_topk', type=int, default=64,
            help="Number of edge features for top-k graph [64]")

    # Loss function parameters
    loss_group = parser.add_argument_group("loss parameters")
    loss_group.add_argument('-w_dist', type=float, default=0.3,
            help="Weight on distd in loss function [0.3]")
    loss_group.add_argument('-w_exp', type=float, default=0.1,
            help="Weight on experimental resolved in loss function [0.1]")
    loss_group.add_argument('-w_str', type=float, default=1.0,
            help="Weight on strd in loss function [1.0]")
    loss_group.add_argument('-w_lddt', type=float, default=0.01,
            help="Weight on predicted lddt loss [0.01]")
    loss_group.add_argument('-w_pae', type=float, default=0.01,
            help="Weight on predicted pae loss [0.01]")
    loss_group.add_argument('-w_all', type=float, default=0.5,
            help="Weight on MSA masked token prediction loss [0.5]")
    loss_group.add_argument('-w_aa', type=float, default=1.0,
            help="Weight on MSA masked token prediction loss [1.0]")
    loss_group.add_argument('-w_blen', type=float, default=0.0,
            help="Weight on predicted blen loss [0.0]")
    loss_group.add_argument('-w_bang', type=float, default=0.0,
            help="Weight on predicted bang loss [0.0]")
    loss_group.add_argument('-w_lj', type=float, default=0.0,
            help="Weight on lj loss [0.0]")
    loss_group.add_argument('-w_hb', type=float, default=0.0,
            help="Weight on hb loss [0.0]")
    loss_group.add_argument('-lj_lin', type=float, default=0.75,
            help="switch from linear to 12-6 for lj potential [0.75]")
    loss_group.add_argument('-use_H', action='store_true', default=False,
            help="consider hydrogens for lj loss [False]")

    # parse arguments
    args = parser.parse_args()

    # Setup dataloader parameters:
    loader_param = data_loader.set_data_loader_params(args)

    # make dictionary for each parameters
    trunk_param = {}
    for param in TRUNK_PARAMS:
        trunk_param[param] = getattr(args, param)
    SE3_param_full = {}
    SE3_param_topk = {}
    for param in SE3_PARAMS:
        if hasattr(args, param):
            if "full" in param:
                SE3_param_full[param[:-5]] = getattr(args, param)
            elif "topk" in param:
                SE3_param_topk[param[:-5]] = getattr(args, param)
            else: # common arguments
                SE3_param_full[param] = getattr(args, param)
                SE3_param_topk[param] = getattr(args, param)
    trunk_param['SE3_param_full'] = SE3_param_full
    trunk_param['SE3_param_topk'] = SE3_param_topk
    loss_param = {}
    for param in ['w_dist', 'w_str', 'w_all', 'w_aa', 'w_lddt', 'w_pae', 'w_blen', 'w_bang', 'w_lj', 'w_hb', 'lj_lin', 'use_H']:
        loss_param[param] = getattr(args, param)

    return args, trunk_param, loader_param, loss_param
