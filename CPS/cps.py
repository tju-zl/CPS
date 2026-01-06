import torch
import torch.nn.functional as F
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .model import CPSModel, ZINBLoss


class CPSTrainer:
    def __init__(self, args):
        # system setting
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        # model and opt
        self.model = CPSModel(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, 
                                          weight_decay=args.weight_decay)
        # self.optimizer_inr = torch.optim.Adam(self.model.student.parameters(), lr=args.lr, 
        #                                   weight_decay=args.weight_decay)
        
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
            # self.optimizer_inr.zero_grad()
            results = self.model(pos, x, edge_index, return_attn=False)
            losses = self.compute_losses(results, y, recon_weight=[0.5, 0.5], verbose=False)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            # self.optimizer_inr.zero_grad()
            
            # Record training loss
            train_total_loss = losses['total'].item()
            loss_history['train_total'].append(train_total_loss)
            
            # Record each loss component
            if 'recon_teacher' in losses:
                loss_history['train_recon_teacher'].append(losses['recon_teacher'].item())
            if 'recon_student' in losses:
                loss_history['train_recon_student'].append(losses['recon_student'].item())
            if 'distill' in losses:
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
        
        
    # infer the spots
    def infer_imputation_spots(self, coords):
        """
        Impute gene expression for spots based on coordinates
        
        Parameters:
            coords: torch.Tensor, shape (N, 2)
                Coordinates of spots to be imputed
        
        Returns:
            imputed_expr: torch.Tensor, shape (N, n_genes)
                Imputed gene expression matrix
        """
        self.model.eval()
        with torch.no_grad():
            # Move coordinates to device
            coords = coords.to(self.device)
            
            # Generate latent representation using student network
            z_student = self.model.student(coords)
            
            # Generate gene expression using decoder
            imputed_expr = self.model.decoder(z_student)
            
            return imputed_expr.cpu()
        
    def infer_imputation_genes(self, pyg_data):
        """
        Impute gene expression (when some genes are masked)
        
        Parameters:
            pyg_data: Data object
                Graph data containing partially masked gene expression, should include:
                - x: gene expression matrix (masked)
                - pos: coordinates
                - edge_index: graph structure (if prep_scale=False)
        
        Returns:
            imputed_expr: torch.Tensor, shape (N, n_genes)
                Complete gene expression matrix (after imputation)
        """
        self.model.eval()
        with torch.no_grad():
            # Extract data and move to device
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos.to(self.device)
            
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            # Generate latent representation using teacher network
            z_teacher, _ = self.model.teacher(x, edge_index, return_attn=False)
            
            # Generate complete gene expression using decoder
            imputed_expr = self.model.decoder(z_teacher)
            
            return imputed_expr.cpu()
    
    # infer the atten scores
    def interpret_attn_scores(self, pyg_data, return_fig=False):
        """
        Interpret attention scores and visualize
        
        Parameters:
            pyg_data: Data object
                Data containing gene expression and coordinates
            return_fig: bool
                Whether to return matplotlib figure object
        
        Returns:
            z_teacher: numpy.ndarray, shape (N, latent_dim)
                Latent representation of teacher network
            attn_weights: numpy.ndarray, shape (N, n_scales, n_heads)
                Attention weights
            fig: matplotlib.figure.Figure, optional
                If return_fig=True, returns figure object
        """
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
            
            if return_fig:
                return (z_student.to('cpu').detach().numpy(),
                        z_teacher.to('cpu').detach().numpy(),
                        attn_weights.to('cpu').detach().numpy(),
                        fig)
            else:
                plt.close(fig)
                return (z_teacher.to('cpu').detach().numpy(),
                        attn_weights.to('cpu').detach().numpy())
            
    def compute_loss(self, pred_dict, gene_expr, recon_weight):
        """
        Compute various losses (supports ZINB loss and traditional loss)
        
        Parameters:
            pred_dict: dict
                Dictionary of model prediction results
            gene_expr: torch.Tensor
                True gene expression
            recon_weight: list
                Reconstruction loss weights [teacher_weight, student_weight]
            verbose: bool
                Whether to print loss information
        
        Returns:
            losses: dict
                Dictionary of various losses
        """
        losses = {}
        
        zinb_teacher = ZINBLoss()
        mean_teacher = pred_dict['mean_teacher']
        disp_teacher = pred_dict['disp_teacher']
        pi_teacher = pred_dict['pi_teacher']
        zinb_loss_teacher = zinb_teacher(gene_expr, mean_teacher, disp_teacher, pi_teacher)
        losses['recon_teacher'] = recon_weight[0] * zinb_loss_teacher
        
        
    
    
    def compute_losses(self, pred_dict, gene_expr, recon_weight, verbose=False):
        """
        Compute various losses (supports ZINB loss and traditional loss)
        
        Parameters:
            pred_dict: dict
                Dictionary of model prediction results
            gene_expr: torch.Tensor
                True gene expression
            recon_weight: list
                Reconstruction loss weights [teacher_weight, student_weight]
            verbose: bool
                Whether to print loss information
        
        Returns:
            losses: dict
                Dictionary of various losses
        """
        losses = {}
        
        # Check if using ZINB decoder
        use_zinb = 'mean_teacher' in pred_dict and 'disp_teacher' in pred_dict and 'pi_teacher' in pred_dict
        
        # Teacher network reconstruction loss
        if 'recon_teacher' in pred_dict:
            if use_zinb:
                # ZINB loss
                zinb_loss = ZINBLoss()
                mean_teacher = pred_dict['mean_teacher']
                disp_teacher = pred_dict['disp_teacher']
                pi_teacher = pred_dict['pi_teacher']
                
                zinb_loss_teacher = zinb_loss(gene_expr, mean_teacher, disp_teacher, pi_teacher)
                # Still compute MSE for monitoring
                mse_loss_teacher = F.mse_loss(pred_dict['recon_teacher'], gene_expr)
                # Combined loss: ZINB as main, plus small MSE for training stability

                recon_loss_teacher = 0.5 * zinb_loss_teacher + 0.5 * mse_loss_teacher
                
                losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
                losses['recon_teacher_zinb'] = zinb_loss_teacher
                losses['recon_teacher_mse'] = mse_loss_teacher
            else:
                # Traditional loss (MSE + cosine similarity)
                recon_teacher = pred_dict['recon_teacher']
                # MSE loss
                mse_loss_teacher = F.mse_loss(recon_teacher, gene_expr)
                # Cosine similarity loss (improves Pearson correlation)
                cosine_loss_teacher = 1 - F.cosine_similarity(recon_teacher, gene_expr, dim=-1).mean()
                # Combined loss
                recon_loss_teacher = mse_loss_teacher + 0.1 * cosine_loss_teacher
                losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
                # Record each component
                losses['recon_teacher_mse'] = mse_loss_teacher
                losses['recon_teacher_cosine'] = cosine_loss_teacher
        
        # Student network reconstruction loss
        if 'recon_student' in pred_dict:
            if use_zinb:
                # ZINB loss
                zinb_loss = ZINBLoss()
                mean_student = pred_dict['mean_student']
                disp_student = pred_dict['disp_student']
                pi_student = pred_dict['pi_student']
                
                zinb_loss_student = zinb_loss(gene_expr, mean_student, disp_student, pi_student)
                # Still compute MSE for monitoring
                mse_loss_student = F.mse_loss(pred_dict['recon_student'], gene_expr)
                # Combined loss
                
                recon_loss_student = 0.5 * zinb_loss_student + 0.5 * mse_loss_student
                
                losses['recon_student'] = recon_weight[1] * recon_loss_student
                losses['recon_student_zinb'] = zinb_loss_student
                losses['recon_student_mse'] = mse_loss_student
            else:
                # Traditional loss
                recon_student = pred_dict['recon_student']
                # MSE loss
                mse_loss_student = F.mse_loss(recon_student, gene_expr)
                # Cosine similarity loss
                cosine_loss_student = 1 - F.cosine_similarity(recon_student, gene_expr, dim=-1).mean()
                # Combined loss
                recon_loss_student = mse_loss_student + 0.1 * cosine_loss_student
                losses['recon_student'] = recon_weight[1] * recon_loss_student
                # Record each component
                losses['recon_student_mse'] = mse_loss_student
                losses['recon_student_cosine'] = cosine_loss_student
            
        # Distillation loss
        if 'distill_loss' in pred_dict:
            losses['distill'] = self.args.distill * pred_dict['distill_loss']
        
        # Compute total loss (excluding monitoring metrics)
        exclude_suffixes = ['_mse', '_cosine', '_zinb']
        total_loss = sum([losses[k] for k in losses
                         if not any(k.endswith(suffix) for suffix in exclude_suffixes)])
        losses['total'] = total_loss
        
        # Optional printing
        if verbose:
            # Main losses
            main_losses = {k: v for k, v in losses.items()
                          if not any(k.endswith(suffix) for suffix in exclude_suffixes)}
            loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in main_losses.items()])
            print(f"Losses: {loss_str}")
            
            # Detailed components
            if use_zinb:
                if 'recon_teacher_zinb' in losses:
                    print(f"  Teacher ZINB: {losses['recon_teacher_zinb'].item():.4f}, "
                          f"MSE: {losses['recon_teacher_mse'].item():.4f}")
                if 'recon_student_zinb' in losses:
                    print(f"  Student ZINB: {losses['recon_student_zinb'].item():.4f}, "
                          f"MSE: {losses['recon_student_mse'].item():.4f}")
            else:
                if 'recon_teacher_mse' in losses:
                    print(f"  Teacher MSE: {losses['recon_teacher_mse'].item():.4f}, "
                          f"Cosine: {losses['recon_teacher_cosine'].item():.4f}")
                if 'recon_student_mse' in losses:
                    print(f"  Student MSE: {losses['recon_student_mse'].item():.4f}, "
                          f"Cosine: {losses['recon_student_cosine'].item():.4f}")
        
        return losses
    
    def evaluate_imputation(self, original_data, imputed_data, test_indices=None,
                           mask_pattern=None, output_dir='./results',
                           experiment_name='imputation_evaluation'):
        """
        Evaluate imputation performance, interfaces with compute_imputation_metrics function in utils_analys.py
        
        Parameters:
            original_data: original data
                - For spots imputation: Data object containing all spots
                - For genes imputation: Data object containing complete gene expression
            imputed_data: imputed data
                - For spots imputation: gene expression matrix of imputed spots (torch.Tensor)
                - For genes imputation: complete gene expression matrix after imputation (torch.Tensor)
            test_indices: indices of test spots (used for spots imputation)
            mask_pattern: gene mask pattern (used for genes imputation), boolean matrix
            output_dir: output directory for results
            experiment_name: experiment name
        
        Returns:
            metrics_dict: dictionary containing all computed metrics
        """
        # Import function from utils_analys
        from .utils_analys import compute_imputation_metrics
        
        # Prepare imputed_data as Data object format
        if isinstance(imputed_data, torch.Tensor):
            # If tensor, need to convert to format compatible with original_data
            # Create a new Data object to store imputation results
            import torch_geometric.data as data
            
            # Copy structure of original data
            imputed_pyg_data = data.Data()
            
            # Copy coordinates and edge indices (if exist)
            if hasattr(original_data, 'pos'):
                imputed_pyg_data.pos = original_data.pos.clone()
            if hasattr(original_data, 'edge_index'):
                imputed_pyg_data.edge_index = original_data.edge_index.clone()
            
            # Set imputed gene expression
            imputed_pyg_data.x = imputed_data
            
            # If spots imputation, need to insert imputation results into correct positions
            if test_indices is not None:
                # Create complete expression matrix
                if hasattr(original_data, 'x'):
                    full_expr = original_data.x.clone()
                    # Place imputation results into test spots positions
                    full_expr[test_indices] = imputed_data
                    imputed_pyg_data.x = full_expr
                else:
                    imputed_pyg_data.x = imputed_data
            else:
                imputed_pyg_data.x = imputed_data
        else:
            # If already Data object, use directly
            imputed_pyg_data = imputed_data
        
        # Call compute_imputation_metrics function
        metrics = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_pyg_data,
            test_indices=test_indices,
            mask_pattern=mask_pattern,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        return metrics
    
    def evaluate_spots_imputation(self, test_data, test_indices=None,
                                 output_dir='./results', experiment_name='spots_imputation'):
        """
        Simplified function to evaluate spots imputation performance, using only test data
        
        Parameters:
            test_data: test data (output of spots_perturb), containing true expression and coordinates
            test_indices: test spots indices (indices in original data, not in test_data)
            output_dir: output directory for results
            experiment_name: experiment name
        
        Returns:
            metrics_dict: dictionary of imputation metrics
        """
        # 1. Perform imputation using infer_imputation_spots
        if test_indices is None:
            n_test = len(test_data.pos)
            # If test_indices is None, assume test_data is already test set
            test_indices_in_test_data = list(range(n_test))
        else:
            n_test = len(test_indices)
            # test_data only contains test spots, so indices should be 0 to n_test-1
            test_indices_in_test_data = list(range(n_test))
        
        print(f"Performing spots imputation, number of test spots: {n_test}")
        
        # Get test coordinates from test_data
        test_coords = test_data.pos
        imputed_expr = self.infer_imputation_spots(test_coords)
        
        # 2. Prepare original data and imputed data
        original_expr = test_data.y
        
        # 3. Use concise metric calculation function
        try:
            from .metrics import compute_spots_imputation_metrics, print_metrics
        except ImportError:
            # If metrics module doesn't exist, use function from utils_analys
            from .utils_analys import compute_imputation_metrics
            import torch_geometric.data as data
            
            original_test_data = data.Data(x=original_expr, pos=test_coords)
            imputed_test_data = data.Data(x=imputed_expr, pos=test_coords)
            
            metrics = compute_imputation_metrics(
                original_data=original_test_data,
                imputed_data=imputed_test_data,
                test_indices=None,
                output_dir=output_dir,
                experiment_name=experiment_name,
                compute_full_data=False
            )
        else:
            # Use concise metric calculation
            # Note: test_data only contains test spots, so use test_indices_in_test_data
            metrics = compute_spots_imputation_metrics(
                original_data=test_data,
                imputed_data=imputed_expr,
                test_indices=test_indices_in_test_data,
                
            )
            
            # Print metrics
            print_metrics(metrics, title=f"Spots imputation metrics - {experiment_name}")
            
            # Save metrics to file
            if output_dir:
                import os
                import json
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
                with open(json_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Metrics saved to: {json_path}")
        
        return metrics
    
    def evaluate_spots_imputation_from_perturb(self, original_data, perturb_result,
                                              output_dir='./results', experiment_name='spots_imputation'):
        """
        Evaluate spots imputation performance directly from output tuple of spots_perturb
        
        Parameters:
            original_data: original Data object containing all spots
            perturb_result: tuple returned by spots_perturb (train_data, test_data, train_indices, test_indices)
            output_dir: output directory for results
            experiment_name: experiment name
        
        Returns:
            metrics_dict: dictionary of imputation metrics
        """
        # Unpack perturb_result
        train_data, test_data, train_indices, test_indices = perturb_result
        
        # Call complete evaluation function
        return self.evaluate_spots_imputation(
            original_data=original_data,
            train_data=train_data,
            test_data=test_data,
            train_indices=train_indices,
            test_indices=test_indices,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
    
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