import argparse
import random
import numpy as np
import torch


def config():
    parser = argparse.ArgumentParser('Configuration File of CPS')
    
    # system configuration
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', type=int, default=2025)

    # data preparation
    parser.add_argument('--radius', default=150, type=int, help='visium=150, hd=2.5')
    parser.add_argument('--clusters', default=7, type=int, help='spatial domains')
    parser.add_argument('--max_neighbors', default=6, type=int, help='nearest neighbors')
    parser.add_argument('--n_spot', default=0, type=int, help='update when read data.')
    parser.add_argument('--hvgs', default=3000, type=int)
    parser.add_argument('--coord_dim', default=2, type=int, help='default 2 for single slice')
    parser.add_argument('--self_loops', default=True, action='store_true')
    parser.add_argument('--flow', type=str, default='source_to_target')
    parser.add_argument('--prep_scale', default=True, action='store_true', help='use pred multi-scale features')

    # model parameters
    parser.add_argument('--decoder', type=str, default='MLP', help='ZINB, NB or MLP')
    parser.add_argument('--latent_dim', type=int, default=64, help='dim of latent')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--k_list', type=list, default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--inr_latent', type=list, default=[256,256,256])
    parser.add_argument('--decoder_latent', type=list, default=[256,512])
    parser.add_argument('--freq', type=int, default=32, help='dim of position encoding')
    parser.add_argument('--distill', type=float, default=0.5)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=10.0)

    # training control
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # analysis configuration
    parser.add_argument('--visual', default=True, action='store_true')

    return parser


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False