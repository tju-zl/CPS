import torch
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .model import CPSModel
from .utils_metrics import *
import os
import json


class CPSTrainer:
    def __init__(self, args, input_dim=None):
        # system setting
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        # model and opt
        self.model = CPSModel(args, in_dim=input_dim).to(self.device)
        self.optimizer_t = torch.optim.Adam(list(self.model.teacher.parameters())+
                                            list(self.model.decoder.parameters()), 
                                            lr=args.lr, 
                                          weight_decay=args.weight_decay)
        self.optimizer_s = torch.optim.Adam(self.model.student.parameters(), 
                                            lr=args.lr, 
                                            weight_decay=args.weight_decay*0.1)
        
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
        library_size = y.sum(1, keepdim=True)
        pos = pyg_data.pos.to(self.device)
        edge_index = pyg_data.edge_index.to(self.device)
        
        # Loss history recording
        loss_history = {
            'train_total': [],
            'train_recon_teacher': [],
            'train_recon_student': [],
            'train_distill': [],
        }
        
        for epoch in tq.tqdm(range(self.args.max_epoch)):
            # teacher training
            self.optimizer_t.zero_grad()
            z_t, mean_t, attn_weights, rec_loss_t = self.model.teacher_forward(x, edge_index, y, library_size)
            teacher_loss = rec_loss_t
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer_t.step()
            
            # student training
            self.optimizer_t.zero_grad()
            self.optimizer_s.zero_grad()
            z_s, mean_s, pid_loss, rec_loss_s = self.model.student_forward(pos, z_t, y, library_size)
            student_loss = self.args.distill*pid_loss + rec_loss_s
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer_s.step()
            
            # Record training loss
            train_total_loss = (teacher_loss + student_loss).item()
            loss_history['train_total'].append(train_total_loss)
            loss_history['train_recon_teacher'].append(rec_loss_t.item())
            loss_history['train_recon_student'].append(rec_loss_s.item())
            loss_history['train_distill'].append(pid_loss.item())
            
            # Print training information (no validation)
            if verbose and (epoch % print_every == 0 or epoch == self.args.max_epoch - 1):
                # Build compact print format
                print(f"\nEpoch {epoch:3d}: ", end="")
                # Training loss
                train_str = f"Train[Total:{train_total_loss:.4f}"
                train_str += f", T:{rec_loss_t.item():.4f}"
                train_str += f", S:{rec_loss_s.item():.4f}"
                train_str += f", D:{pid_loss.item():.4f}"
                train_str += "]"
                print(train_str)
    
    def efficient_fit(self, pyg_data, verbose=True, print_every=10):
        self.model.train()
        loader = DataLoader(pyg_data, batch_size=self.args.batch_size, shuffle=True)

        for epoch in tq.tqdm(range(self.args.max_epoch)):
            epoch_loss_total = 0
            epoch_loss_t = 0
            epoch_loss_s = 0
            epoch_loss_d = 0
            num_batches = 0

            for batch in loader:
                x = batch.x.to(self.device)
                y = batch.y.to(self.device)
                pos = batch.pos.to(self.device)
                library_size = y.sum(1, keepdim=True)
                edge_index = None
                
                # --- Teacher Training ---
                self.optimizer_t.zero_grad()
                z_t, mean_t, attn_weights, rec_loss_t = self.model.teacher_forward(x, edge_index, y, library_size)
                teacher_loss = rec_loss_t
                teacher_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer_t.step()
                
                # --- Student Training ---
                self.optimizer_t.zero_grad()
                self.optimizer_s.zero_grad()
                z_s, mean_s, pid_loss, rec_loss_s = self.model.student_forward(pos, z_t, y, library_size)
                student_loss = self.args.distill * pid_loss + rec_loss_s
                student_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer_s.step()
                
                epoch_loss_total += (teacher_loss + student_loss).item()
                epoch_loss_t += rec_loss_t.item()
                epoch_loss_s += rec_loss_s.item()
                epoch_loss_d += pid_loss.item()
                num_batches += 1
                
            if verbose and (epoch % print_every == 0 or epoch == self.args.max_epoch - 1):
                avg_total = epoch_loss_total / num_batches
                avg_t = epoch_loss_t / num_batches
                avg_s = epoch_loss_s / num_batches
                avg_d = epoch_loss_d / num_batches
                
                print(f"\nEpoch {epoch:3d}: ", end="")
                train_str = f"Train[Total:{avg_total:.4f}"
                train_str += f", T:{avg_t:.4f}"
                train_str += f", S:{avg_s:.4f}"
                train_str += f", D:{avg_d:.4f}]"
                print(train_str)
     
    # infer position with INR
    def infer_postion(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            # Move coordinates to device
            y = pyg_data.y.to(self.device)
            library_size = y.sum(1, keepdim=True)
            coords = pyg_data.pos.to(self.device)
            # Generate latent representation using student network
            z_student = self.model.student(coords)
            # Generate gene expression using decoder
            imputed_expr, _ = self.model.decoder(z_student, library_size)
            
            return imputed_expr, z_student
    
    # infer graph with multi-scale graph attention teacher
    def infer_graph(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            # Extract data and move to device
            x = pyg_data.x.to(self.device)
            y = pyg_data.y.to(self.device)
            library_size = y.sum(1, keepdim=True)
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            z_t, mean_t, attn_weights, _ = self.model.teacher_forward(x, edge_index, y, library_size)

            return z_t, mean_t, attn_weights
    
    # interpret the attention scores
    def interpret_attn_scores(self, pyg_data):
        self.model.eval()
        with torch.no_grad():
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos.to(self.device)
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            z_teacher, attn_weights, _ = self.model.teacher(x, edge_index)
            z_student = self.model.student(pos)

            return (z_teacher.to('cpu').detach().numpy(), 
                    z_student.to('cpu').detach().numpy(),
                    attn_weights.to('cpu').detach().numpy())

    # evaluation
    def evaluate_spots_imputation(self, test_data, 
                                  output_dir='./results', 
                                  experiment_name='spots_imputation',
                                  use_log1p=False):
        test_expr = test_data.y
        n_test = len(test_data.pos)
        print(f"Performing spots imputation, number of test spots: {n_test}")
        imputed_expr, _ = self.infer_postion(test_data)
        
        # Prepare original data and imputed data
        metrics = compute_spots_imputation_metrics(test_expr=test_expr, 
                                                   imputed_expr=imputed_expr,
                                                   use_log1p=use_log1p,
                                                   verbose=True)
            
        # Save metrics to file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {json_path}")
        return metrics
    
    def evaluate_genes_imputation(self, test_data, 
                                  mask_pattern, 
                                  use_log1p=False, 
                                  output_dir='./results', 
                                  experiment_name='genes_imputation'):
        test_expr = test_data.y
        if torch.is_tensor(mask_pattern):
            mask_pattern = mask_pattern.detach().cpu().numpy().astype(bool)
        n_masked = np.sum(mask_pattern)
        print(f"Performing genes imputation evaluation...")
        print(f"Total entries: {mask_pattern.size}, Masked entries: {n_masked} ({n_masked/mask_pattern.size:.1%})")
            
        imputed_expr, _ = self.infer_postion(test_data)
        
        metrics = compute_genes_imputation_metrics(test_expr=test_expr, 
                                                   imputed_expr=imputed_expr, 
                                                   mask_pattern=mask_pattern, 
                                                   use_log1p=use_log1p,
                                                   verbose=True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
            
            # Helper to convert numpy/tensor types to python floats for JSON
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=convert_to_serializable)
            print(f"Metrics saved to: {json_path}")
        return metrics