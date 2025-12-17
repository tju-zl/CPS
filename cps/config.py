import argparse

def config():
    parser = argparse.ArgumentParser('Configuration File of SpatialGPT')
    
    # system configuration
    parser.add_argument('--project_name', type=str, default='SpatialGPT')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--version', type=str, default='dev-1.0', help='if dev on, warnings will appear.')

    # data preparation
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--vocab_path', type=str, default='/home/zl/SpatialGRN/Data/default_census_vocab.json')
    parser.add_argument('--srt_resolution', default=150, type=int)
    parser.add_argument('--clusters', default=0, type=int)
    parser.add_argument('--max_neighbors', default=6, type=int)
    parser.add_argument('--n_spot', default=0, type=int, help='update when read data.')
    parser.add_argument('--hvgs', default=1200, type=int)
    parser.add_argument('--mode_rknn', type=str, default='rknn')

    # token setting
    parser.add_argument('--cls_initial', default=True, action='store_true')
    parser.add_argument('--n_gcn', default=4, type=int, help='number of GCN layer')
    parser.add_argument('--n_randwalk', default=50, type=int, help='number of node2vec paths')
    parser.add_argument('--gid_dim', default=45, type=int, help='number of gene embedding dim')
    parser.add_argument('--n_token', default=60697, type=int, help='library size of gene vocab')
    parser.add_argument('--n_randomwalk', default=[5, 15], type=list, help='interval of path distance in random walk')
    parser.add_argument('--p_randomwalk', default=[1, 1.5], type=list, help='list, return of random walk')
    parser.add_argument('--q_randomwalk', default=[1, 0.5], type=list, help='interval in-out of random walk')

    # model parameters
    parser.add_argument('--decoder', type=str, default='NB', help='ZINB, NB or MLP')
    parser.add_argument('--latent_dim', type=int, default=100, help='dim of QKV')
    parser.add_argument('--flow', type=str, default='source_to_target')
    parser.add_argument('--dropout', type=float, default=0.3)
    # parser.add_argument('--att_mask', default=True, action='store_true')

    # Transformer Module parameters
    parser.add_argument('--d_model', type=int, default=100, help='gene id+exp dims')
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--fast', default=True, action='store_true')
    parser.add_argument('--f_dim', type=int, default=100, help='forward dim')

    # training control
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--early_stop', default=True, action='store_true')
    parser.add_argument('--schedule_ratio', type=float, default=0.9)
    parser.add_argument('--wegiht_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.5)
    # parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--amp', default=True, action='store_true', help='Mixed precision')
    
    # output configuration
    parser.add_argument('--log_file', type=str, default='../Log')
    parser.add_argument('--out_file', type=str, default='../Output')

    # analysis configuration
    parser.add_argument('--visual', default=True, action='store_true')

    return parser