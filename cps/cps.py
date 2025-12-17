import torch
from .model import CPSModel


class CPSTrainer:
    def __init__(self, args, adata, model):
        self.device = torch.device
        
        self.model = CPSModel()