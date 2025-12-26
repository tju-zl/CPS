import torch
import torch.nn.functional as F
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    def infer_imputation_spots(self, coords):
        self.model.eval()
        with torch.no_grad():
            pass
        
    def infer_imputation_genes(self, pyg_data):
        pass
    
    # infer the atten scores
    def interpret_attn_scores(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            z_teacher, attn_weights = self.model.teacher(x, edge_index, return_weights=True)
            attn_avg_heads = attn_weights.mean(dim=-1).cpu().numpy()
            n_scales = len(self.args.k_list)
            fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 5))
            if n_scales == 1: axes = [axes]
    
            for i, k in enumerate(self.args.k_list):
                ax = axes[i]
                sc = ax.scatter(pos[:, 0], pos[:, 1], 
                                c=attn_avg_heads[:, i], 
                                cmap='viridis', s=10, alpha=0.8)
                ax.set_title(f"Attention to Scale K={k}")
                ax.axis('off')
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                
            plt.suptitle("Spatial Attention", fontsize=16)
            plt.tight_layout()
            
            return z_teacher.to('cpu').detach().numpy(), attn_weights.to('cpu').detach().numpy()
            
    
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
        print(sum([losses[k] for k in losses]).item())
        total_loss = sum([losses[k] for k in losses])
        losses['total'] = total_loss
        
        return losses