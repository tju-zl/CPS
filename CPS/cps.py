import torch
import torch.nn.functional as F
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .model import CPSModel, ZINBLoss
from .utils_metrics import *
import os
import json


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
    def fit(self, pyg_data, verbose=True, print_every=10):
        """
        Train CPS model
        
        Parameters:
            pyg_data: Data object, training data
            verbose: bool, whether to print training information
            print_every: int, print loss information every N epochs
        """
        self.model.train()
        x = pyg_data.x.to(self.device)
        y = pyg_data.y.to(self.device)
        pos = pyg_data.pos.to(self.device)
        if not self.args.prep_scale:
            edge_index = pyg_data.edge_index.to(self.device)
        else:
            edge_index = None
        
        # Loss history recording
        loss_history = {
            'train_total': [],
            'train_recon_teacher': [],
            'train_recon_student': [],
            'train_distill': [],
        }
        
        for epoch in tq.tqdm(range(self.args.max_epoch)):
            self.model.train()
            self.optimizer.zero_grad()
            results = self.model(pos, x, edge_index, return_attn=False)
            losses = self.compute_losses(results, y, recon_weight=[0.5, 0.5])
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            
            # Record training loss
            train_total_loss = losses['total'].item()
            loss_history['train_total'].append(train_total_loss)
            loss_history['train_recon_teacher'].append(losses['recon_teacher'].item())
            loss_history['train_recon_student'].append(losses['recon_student'].item())
            loss_history['train_distill'].append(losses['distill'].item())
            
            # Print training information (no validation)
            if verbose and (epoch % print_every == 0 or epoch == self.args.max_epoch - 1):
                # Build compact print format
                print(f"\nEpoch {epoch:3d}: ", end="")
                # Training loss
                train_str = f"Train[Total:{train_total_loss:.4f}"
                if 'recon_teacher' in losses:
                    train_str += f", T:{losses['recon_teacher'].item():.4f}"
                if 'recon_student' in losses:
                    train_str += f", S:{losses['recon_student'].item():.4f}"
                if 'distill' in losses:
                    train_str += f", D:{losses['distill'].item():.4f}"
                train_str += "]"
                print(train_str)
    
    def compute_losses(self, pred_dict, gene_expr, recon_weight):
        losses = {}
        
        # Teacher network reconstruction loss
        zinb_t_loss = ZINBLoss()
        mean_teacher = pred_dict['mean_teacher']
        disp_teacher = pred_dict['disp_teacher']
        pi_teacher = pred_dict['pi_teacher']
        zinb_loss_teacher = zinb_t_loss(gene_expr, mean_teacher, disp_teacher, pi_teacher)
        mse_loss_teacher = F.mse_loss(pred_dict['recon_teacher'], gene_expr)
        # Combined loss: ZINB plus MSE for training stability
        recon_loss_teacher = 0.5 * zinb_loss_teacher + 0.5 * mse_loss_teacher
        losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
        # losses['recon_teacher_zinb'] = zinb_loss_teacher
        # losses['recon_teacher_mse'] = mse_loss_teacher
            
        # Student network reconstruction loss
        zinb_s_loss = ZINBLoss()
        mean_student = pred_dict['mean_student']
        disp_student = pred_dict['disp_student']
        pi_student = pred_dict['pi_student']
        zinb_loss_student = zinb_s_loss(gene_expr, mean_student, disp_student, pi_student)
        mse_loss_student = F.mse_loss(pred_dict['recon_student'], gene_expr)
        recon_loss_student = 0.5 * zinb_loss_student + 0.5 * mse_loss_student
        losses['recon_student'] = recon_weight[1] * recon_loss_student
        # losses['recon_student_zinb'] = zinb_loss_student
        # losses['recon_student_mse'] = mse_loss_student

        # Distillation loss
        losses['distill'] = self.args.distill * pred_dict['distill_loss']
        
        # Compute total loss
        total_loss = sum([losses[k] for k in losses])
        losses['total'] = total_loss

        return losses
     
    # infer position with INR
    def infer_postion(self, coords):
        self.model.eval()
        with torch.no_grad():
            # Move coordinates to device
            coords = coords.to(self.device)
            
            # Generate latent representation using student network
            z_student = self.model.student(coords)
            
            # Generate gene expression using decoder
            imputed_expr = self.model.decoder(z_student)
            
            return imputed_expr
    
    # infer graph with multi-scale graph attention teacher
    def infer_graph(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            # Extract data and move to device
            x = pyg_data.x.to(self.device)
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            # Generate latent representation using teacher network
            z_teacher, _ = self.model.teacher(x, edge_index, return_attn=False)
            
            # Generate complete gene expression using decoder
            imputed_expr = self.model.decoder(z_teacher)
            
            return imputed_expr
    
    # interpret the attention scores
    def interpret_attn_scores(self, pyg_data, verbose=False):
        self.model.eval()
        with torch.no_grad():
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos.to(self.device)
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            z_teacher, attn_weights = self.model.teacher(x, edge_index, return_weights=True)
            z_student = self.model.student(pos)
            
            attn_avg_heads = attn_weights.mean(dim=-1).cpu().numpy()
            pos_cpu = pos.cpu().numpy()
            
            n_scales = len(self.args.k_list)
            fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 5))
            if n_scales == 1:
                axes = [axes]
    
            for i, k in enumerate(self.args.k_list):
                ax = axes[i]
                sc = ax.scatter(pos_cpu[:, 0], pos_cpu[:, 1],
                                c=attn_avg_heads[:, i],
                                cmap='viridis', s=10, alpha=0.8)
                ax.set_title(f"Attention to Scale K={k}")
                ax.axis('off')
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                
            plt.suptitle("Spatial Attention", fontsize=16)
            plt.tight_layout()
            
            if verbose:
                return (z_student.to('cpu').detach().numpy(),
                        z_teacher.to('cpu').detach().numpy(),
                        attn_weights.to('cpu').detach().numpy(),
                        fig)
            else:
                plt.close(fig)
                return (z_teacher.to('cpu').detach().numpy(),
                        attn_weights.to('cpu').detach().numpy())
    
    # evaluation
    def evaluate_spots_imputation(self, test_data, output_dir='./results', experiment_name='spots_imputation'):
        test_coords = test_data.pos
        test_expr = test_data.y
        n_test = len(test_data.pos)
        print(f"Performing spots imputation, number of test spots: {n_test}")
        test_coords = test_data.pos
        imputed_expr = self.infer_postion(test_coords)
        
        # Prepare original data and imputed data
        metrics = compute_spots_imputation_metrics(test_expr=test_expr, imputed_expr=imputed_expr)
            
        # Save metrics to file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {json_path}")
        return metrics

    
    def evaluate_genes_imputation(self, original_data, mask_pattern,
                                 output_dir='./results', experiment_name='genes_imputation'):
        """
        Simplified function to evaluate genes imputation performance
        
        Parameters:
            original_data: original Data object containing complete gene expression
            mask_pattern: gene mask pattern, boolean matrix
            output_dir: output directory for results
            experiment_name: experiment name
        
        Returns:
            metrics_dict: dictionary of imputation metrics
        """
        # 1. Create masked data
        import copy
        import torch
        
        masked_data = copy.deepcopy(original_data)
        
        # Apply mask
        if isinstance(mask_pattern, torch.Tensor):
            mask_tensor = mask_pattern
        else:
            mask_tensor = torch.tensor(mask_pattern, dtype=torch.bool)
        
        # Set gene expression at masked positions to 0
        masked_data.x = masked_data.x.clone()
        masked_data.x[mask_tensor] = 0
        
        # 2. Perform imputation using infer_imputation_genes
        print(f"Performing genes imputation, number of masked elements: {mask_tensor.sum().item()}")
        imputed_expr = self.infer_imputation_genes(masked_data)
        
        # 3. Directly compute metrics
        from .utils_analys import compute_imputation_metrics
        
        # Create imputed data object
        import torch_geometric.data as data
        
        imputed_data = data.Data(
            x=imputed_expr,
            pos=original_data.pos.clone() if hasattr(original_data, 'pos') else None,
            edge_index=original_data.edge_index.clone() if hasattr(original_data, 'edge_index') else None
        )
        
        # 4. Compute metrics
        metrics = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_data,
            mask_pattern=mask_pattern,
            output_dir=output_dir,
            experiment_name=experiment_name,
            compute_full_data=False  # Only compute metrics for masked positions
        )
        
        return metrics