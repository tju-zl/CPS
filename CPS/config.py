import argparse
import random
import numpy as np
import torch


def config():
    parser = argparse.ArgumentParser('Configuration File of CPS')
    
    # system configuration
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', type=int, default=42)

    # data preparation
    parser.add_argument('--radius', default=150, type=int, help='visium=150, hd=2.5, stereo-seq=15')
    parser.add_argument('--clusters', default=7, type=int, help='spatial domains')
    parser.add_argument('--max_neighbors', default=6, type=int, help='num of nearest neighbors')
    parser.add_argument('--n_spot', default=0, type=int, help='auto update when read data.')
    parser.add_argument('--hvgs', default=3000, type=int)
    parser.add_argument('--coord_dim', default=2, type=int, help='dim of coordination')
    parser.add_argument('--self_loops', default=True, action='store_true')
    parser.add_argument('--flow', type=str, default='source_to_target')
    parser.add_argument('--prep_scale', default=True, action='store_true', help='use pred multi-scale features')

    # model parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='dim of latent')
    # parser.add_argument('--embedd_dim', type=int, default=512, help='dim of embedd')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--k_list', nargs='+', type=int, default=[0,1,2,3,4,5,6,7]) 
    parser.add_argument('--inr_latent', nargs='+', type=int, default=[256,256,256])
    parser.add_argument('--decoder_latent', nargs='+', type=int, default=[256, 512, 1024])
    parser.add_argument('--freq', type=int, default=32, help='dim of position encoding')
    parser.add_argument('--distill', type=float, default=1.0)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--er_w', type=float, default=0.05)

    # training control
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    return parser


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False