import torch
from .model import CPSModel


class CPSTrainer:
    def __init__(self, args, adata):
        # system setting
        self.args = args
        self.data_mode = args.resolved
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # data to tensor
        self.features = torch.FloatTensor(adata.obsm['raw_feature'].copy()).to(self.device)
        args.hvgs = adata.n_var
        
        # model and opt
        self.model = CPSModel(num_genes=args.hvgs,
                              latent_dim=args.latent_dim,
                              teacher_k_list=args.k_list,
                              num_frequencies=args.freq,
                              lambda_distill=args.distill)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=args.lr, weight_decay=args.weight_decay)
        
    # train the model
    def fit(self, visual):
        self.model.train()
        
    # infer the results
    def infer(self, coords):
        pass
    