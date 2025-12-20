import torch
import torch.nn.functional as F
import tqdm.notebook as tq
from .model import CPSModel


class CPSTrainer:
    def __init__(self, args):
        # system setting
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        # model and opt
        self.model = CPSModel(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, 
                                          weight_decay=args.weight_decay)
        
    # train the model
    def fit(self, pyg_data):
        self.model.train()
        x = pyg_data.x.to(self.device)
        pos = pyg_data.pos.to(self.device)
        if not self.args.prep_scale:
            edge_index = pyg_data.edge_index.to(self.device)
        else:
            edge_index = None
        
        for _ in tq.tqdm(range(1, self.args.max_epoch)):
            self.optimizer.zero_grad()
            results = self.model(pos, x, edge_index, return_attn=False)
            losses = self.compute_losses(results, x, recon_weight=[0.5, 0.5])
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        
    # infer the spots
    def infer_spots(self, coords):
        self.model.eval()
        with torch.no_grad():
            pass
    
    # infer the atten scores
    def infer_att_scores(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            pass
    
    def compute_losses(self, pred_dict, gene_expr, recon_weight):
        losses = {}
        if 'recon_teacher' in pred_dict:
            recon_loss_teacher = F.mse_loss(pred_dict['recon_teacher'], gene_expr)
            losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
        
        if 'recon_student' in pred_dict:
            recon_loss_student = F.mse_loss(pred_dict['recon_student'], gene_expr)
            losses['recon_student'] = recon_weight[1] * recon_loss_student
            
        if 'distill_loss' in pred_dict:
            losses['distill'] = self.args.distill * pred_dict['distill_loss']

        total_loss = torch.sum([losses[k] for k in losses])
        losses['total'] = total_loss
        
        return losses