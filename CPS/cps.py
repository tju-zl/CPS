import torch
from .model import CPSModel


class CPSTrainer:
    def __init__(self, args, adata):
        
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        
        