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
    def fit(self, pyg_data, val_data=None, patience=20, verbose=True, print_every=10):
        """
        训练CPS模型
        
        参数:
            pyg_data: Data object, 训练数据
            val_data: Data object, 可选，验证数据
            patience: int, 早停耐心值
            verbose: bool, 是否打印训练信息
            print_every: int, 每隔多少epoch打印一次损失信息
        """
        self.model.train()
        x = pyg_data.x.to(self.device)
        y = pyg_data.y.to(self.device)
        pos = pyg_data.pos.to(self.device)
        if not self.args.prep_scale:
            edge_index = pyg_data.edge_index.to(self.device)
        else:
            edge_index = None
        
        # 初始化学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 损失历史记录
        loss_history = {
            'train_total': [],
            'train_recon_teacher': [],
            'train_recon_student': [],
            'train_distill': [],
            'val_total': [],
            'val_recon_teacher': [],
            'val_recon_student': [],
            'val_distill': []
        }
        
        for epoch in tq.tqdm(range(self.args.max_epoch)):
            self.model.train()
            self.optimizer.zero_grad()
            results = self.model(pos, x, edge_index, return_attn=False)
            losses = self.compute_losses(results, y, recon_weight=[0.5, 0.5], verbose=False)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            
            # 记录训练损失
            train_total_loss = losses['total'].item()
            loss_history['train_total'].append(train_total_loss)
            
            # 记录各个损失分量
            if 'recon_teacher' in losses:
                loss_history['train_recon_teacher'].append(losses['recon_teacher'].item())
            if 'recon_student' in losses:
                loss_history['train_recon_student'].append(losses['recon_student'].item())
            if 'distill' in losses:
                loss_history['train_distill'].append(losses['distill'].item())
            
            # 验证
            if val_data is not None:
                val_losses = self.validate(val_data, return_all_losses=True)
                val_total_loss = val_losses['total']
                loss_history['val_total'].append(val_total_loss)
                
                # 记录验证损失的各个分量
                if 'recon_teacher' in val_losses:
                    loss_history['val_recon_teacher'].append(val_losses['recon_teacher'])
                if 'recon_student' in val_losses:
                    loss_history['val_recon_student'].append(val_losses['recon_student'])
                if 'distill' in val_losses:
                    loss_history['val_distill'].append(val_losses['distill'])
                
                # 学习率调度
                scheduler.step(val_total_loss)
                
                # 早停检查
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
                
                # 打印训练信息
                if verbose and (epoch % print_every == 0 or epoch == self.args.max_epoch - 1):
                    # 构建紧凑的打印格式
                    print(f"\nEpoch {epoch:3d}: ", end="")
                    
                    # 训练损失
                    train_str = f"Train[Total:{train_total_loss:.4f}"
                    if 'recon_teacher' in losses:
                        train_str += f", T:{losses['recon_teacher'].item():.4f}"
                    if 'recon_student' in losses:
                        train_str += f", S:{losses['recon_student'].item():.4f}"
                    if 'distill' in losses:
                        train_str += f", D:{losses['distill'].item():.4f}"
                    train_str += "]"
                    
                    # 验证损失
                    val_str = f"Val[Total:{val_total_loss:.4f}"
                    if 'recon_teacher' in val_losses:
                        val_str += f", T:{val_losses['recon_teacher']:.4f}"
                    if 'recon_student' in val_losses:
                        val_str += f", S:{val_losses['recon_student']:.4f}"
                    if 'distill' in val_losses:
                        val_str += f", D:{val_losses['distill']:.4f}"
                    val_str += "]"
                    
                    print(f"{train_str} | {val_str}")
            else:
                # 打印训练信息（无验证）
                if verbose and (epoch % print_every == 0 or epoch == self.args.max_epoch - 1):
                    # 构建紧凑的打印格式
                    print(f"\nEpoch {epoch:3d}: ", end="")
                    
                    # 训练损失
                    train_str = f"Train[Total:{train_total_loss:.4f}"
                    if 'recon_teacher' in losses:
                        train_str += f", T:{losses['recon_teacher'].item():.4f}"
                    if 'recon_student' in losses:
                        train_str += f", S:{losses['recon_student'].item():.4f}"
                    if 'distill' in losses:
                        train_str += f", D:{losses['distill'].item():.4f}"
                    train_str += "]"
                    
                    print(train_str)
        
        # 恢复最佳模型
        if val_data is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        # 返回损失历史
        # return loss_history
    
    def validate(self, pyg_data, return_all_losses=False):
        """
        验证模型
        
        参数:
            pyg_data: Data object, 验证数据
            return_all_losses: bool, 是否返回所有损失分量
        
        返回:
            如果return_all_losses=True: 返回损失字典
            否则: 返回总损失值
        """
        self.model.eval()
        with torch.no_grad():
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos.to(self.device)
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            results = self.model(pos, x, edge_index, return_attn=False)
            losses = self.compute_losses(results, x, recon_weight=[0.5, 0.5], verbose=False)
            
            if return_all_losses:
                # 转换为标量值
                loss_dict = {k: v.item() for k, v in losses.items()}
                return loss_dict
            else:
                return losses['total'].item()
        
    # infer the spots
    def infer_imputation_spots(self, coords):
        """
        根据坐标填补spots的基因表达
        
        参数:
            coords: torch.Tensor, shape (N, 2)
                需要填补的spots坐标
        
        返回:
            imputed_expr: torch.Tensor, shape (N, n_genes)
                填补后的基因表达矩阵
        """
        self.model.eval()
        with torch.no_grad():
            # 将坐标移动到设备上
            coords = coords.to(self.device)
            
            # 使用学生网络生成潜在表示
            z_student = self.model.student(coords)
            
            # 使用解码器生成基因表达
            imputed_expr = self.model.decoder(z_student)
            
            return imputed_expr.cpu()
        
    def infer_imputation_genes(self, pyg_data):
        """
        填补基因表达（部分基因被mask的情况）
        
        参数:
            pyg_data: Data object
                包含部分mask基因表达的图数据，需要包含：
                - x: 基因表达矩阵（已mask）
                - pos: 坐标
                - edge_index: 图结构（如果prep_scale=False）
        
        返回:
            imputed_expr: torch.Tensor, shape (N, n_genes)
                完整的基因表达矩阵（填补后的）
        """
        self.model.eval()
        with torch.no_grad():
            # 提取数据并移动到设备
            x = pyg_data.x.to(self.device)
            pos = pyg_data.pos.to(self.device)
            
            if not self.args.prep_scale:
                edge_index = pyg_data.edge_index.to(self.device)
            else:
                edge_index = None
            
            # 使用教师网络生成潜在表示
            z_teacher, _ = self.model.teacher(x, edge_index, return_attn=False)
            
            # 使用解码器生成完整的基因表达
            imputed_expr = self.model.decoder(z_teacher)
            
            return imputed_expr.cpu()
    
    # infer the atten scores
    def interpret_attn_scores(self, pyg_data, return_fig=False):
        """
        解释注意力分数并可视化
        
        参数:
            pyg_data: Data object
                包含基因表达和坐标的数据
            return_fig: bool
                是否返回matplotlib图形对象
        
        返回:
            z_teacher: numpy.ndarray, shape (N, latent_dim)
                教师网络的潜在表示
            attn_weights: numpy.ndarray, shape (N, n_scales, n_heads)
                注意力权重
            fig: matplotlib.figure.Figure, 可选
                如果return_fig=True，返回图形对象
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
            
    
    def compute_losses(self, pred_dict, gene_expr, recon_weight, verbose=False):
        """
        计算各种损失（支持ZINB损失和传统损失）
        
        参数:
            pred_dict: dict
                模型预测结果字典
            gene_expr: torch.Tensor
                真实的基因表达
            recon_weight: list
                重建损失的权重 [teacher_weight, student_weight]
            verbose: bool
                是否打印损失信息
        
        返回:
            losses: dict
                各种损失的字典
        """
        losses = {}
        
        # 检查是否使用ZINB解码器
        use_zinb = 'mean_teacher' in pred_dict and 'disp_teacher' in pred_dict and 'pi_teacher' in pred_dict
        
        # 教师网络重建损失
        if 'recon_teacher' in pred_dict:
            if use_zinb:
                # ZINB损失
                from .model import ZINBLoss
                zinb_loss = ZINBLoss()
                mean_teacher = pred_dict['mean_teacher']
                disp_teacher = pred_dict['disp_teacher']
                pi_teacher = pred_dict['pi_teacher']
                
                zinb_loss_teacher = zinb_loss(gene_expr, mean_teacher, disp_teacher, pi_teacher)
                # 仍然计算MSE用于监控
                mse_loss_teacher = F.mse_loss(pred_dict['recon_teacher'], gene_expr)
                # 组合损失：ZINB为主，加上少量MSE稳定训练
                recon_loss_teacher = zinb_loss_teacher + 0.1 * mse_loss_teacher
                
                losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
                losses['recon_teacher_zinb'] = zinb_loss_teacher
                losses['recon_teacher_mse'] = mse_loss_teacher
            else:
                # 传统损失（MSE + 余弦相似度）
                recon_teacher = pred_dict['recon_teacher']
                # MSE损失
                mse_loss_teacher = F.mse_loss(recon_teacher, gene_expr)
                # 余弦相似度损失（提升皮尔森相关性）
                cosine_loss_teacher = 1 - F.cosine_similarity(recon_teacher, gene_expr, dim=-1).mean()
                # 组合损失
                recon_loss_teacher = mse_loss_teacher + 0.1 * cosine_loss_teacher
                losses['recon_teacher'] = recon_weight[0] * recon_loss_teacher
                # 记录各个分量
                losses['recon_teacher_mse'] = mse_loss_teacher
                losses['recon_teacher_cosine'] = cosine_loss_teacher
        
        # 学生网络重建损失
        if 'recon_student' in pred_dict:
            if use_zinb:
                # ZINB损失
                from .model import ZINBLoss
                zinb_loss = ZINBLoss()
                mean_student = pred_dict['mean_student']
                disp_student = pred_dict['disp_student']
                pi_student = pred_dict['pi_student']
                
                zinb_loss_student = zinb_loss(gene_expr, mean_student, disp_student, pi_student)
                # 仍然计算MSE用于监控
                mse_loss_student = F.mse_loss(pred_dict['recon_student'], gene_expr)
                # 组合损失
                recon_loss_student = zinb_loss_student + 0.1 * mse_loss_student
                
                losses['recon_student'] = recon_weight[1] * recon_loss_student
                losses['recon_student_zinb'] = zinb_loss_student
                losses['recon_student_mse'] = mse_loss_student
            else:
                # 传统损失
                recon_student = pred_dict['recon_student']
                # MSE损失
                mse_loss_student = F.mse_loss(recon_student, gene_expr)
                # 余弦相似度损失
                cosine_loss_student = 1 - F.cosine_similarity(recon_student, gene_expr, dim=-1).mean()
                # 组合损失
                recon_loss_student = mse_loss_student + 0.1 * cosine_loss_student
                losses['recon_student'] = recon_weight[1] * recon_loss_student
                # 记录各个分量
                losses['recon_student_mse'] = mse_loss_student
                losses['recon_student_cosine'] = cosine_loss_student
            
        # 蒸馏损失
        if 'distill_loss' in pred_dict:
            losses['distill'] = self.args.distill * pred_dict['distill_loss']
        
        # 计算总损失（排除监控指标）
        exclude_suffixes = ['_mse', '_cosine', '_zinb']
        total_loss = sum([losses[k] for k in losses
                         if not any(k.endswith(suffix) for suffix in exclude_suffixes)])
        losses['total'] = total_loss
        
        # 可选打印
        if verbose:
            # 主要损失
            main_losses = {k: v for k, v in losses.items()
                          if not any(k.endswith(suffix) for suffix in exclude_suffixes)}
            loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in main_losses.items()])
            print(f"Losses: {loss_str}")
            
            # 详细分量
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
        评估填补性能，对接utils_analys.py中的compute_imputation_metrics函数
        
        参数:
            original_data: 原始数据
                - 如果是spots填补：Data object包含所有spots
                - 如果是genes填补：Data object包含完整基因表达
            imputed_data: 填补后的数据
                - 如果是spots填补：填补spots的基因表达矩阵 (torch.Tensor)
                - 如果是genes填补：填补后的完整基因表达矩阵 (torch.Tensor)
            test_indices: 测试spots的索引（spots填补时使用）
            mask_pattern: 基因掩码模式（genes填补时使用），布尔矩阵
            output_dir: 结果保存目录
            experiment_name: 实验名称
        
        返回:
            metrics_dict: 包含所有计算指标的字典
        """
        # 导入utils_analys中的函数
        from .utils_analys import compute_imputation_metrics
        
        # 准备imputed_data为Data对象格式
        if isinstance(imputed_data, torch.Tensor):
            # 如果是tensor，需要转换为与original_data兼容的格式
            # 创建一个新的Data对象来保存填补结果
            import torch_geometric.data as data
            
            # 复制原始数据的结构
            imputed_pyg_data = data.Data()
            
            # 复制坐标和边索引（如果存在）
            if hasattr(original_data, 'pos'):
                imputed_pyg_data.pos = original_data.pos.clone()
            if hasattr(original_data, 'edge_index'):
                imputed_pyg_data.edge_index = original_data.edge_index.clone()
            
            # 设置填补后的基因表达
            imputed_pyg_data.x = imputed_data
            
            # 如果是spots填补，需要将填补结果插入到正确位置
            if test_indices is not None:
                # 创建完整的表达矩阵
                if hasattr(original_data, 'x'):
                    full_expr = original_data.x.clone()
                    # 将填补结果放入测试spots位置
                    full_expr[test_indices] = imputed_data
                    imputed_pyg_data.x = full_expr
                else:
                    imputed_pyg_data.x = imputed_data
            else:
                imputed_pyg_data.x = imputed_data
        else:
            # 如果已经是Data对象，直接使用
            imputed_pyg_data = imputed_data
        
        # 调用compute_imputation_metrics函数
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
        评估spots填补性能的简化函数，只使用测试数据
        
        参数:
            test_data: 测试数据（spots_perturb输出），包含真实表达和坐标
            test_indices: 测试spots索引（原始数据中的索引，不是test_data中的索引）
            output_dir: 结果保存目录
            experiment_name: 实验名称
        
        返回:
            metrics_dict: 填补指标字典
        """
        # 1. 使用infer_imputation_spots进行填补
        if test_indices is None:
            n_test = len(test_data.pos)
            # 如果test_indices为None，假设test_data已经是测试集
            test_indices_in_test_data = list(range(n_test))
        else:
            n_test = len(test_indices)
            # test_data只包含测试spots，所以索引应该是0到n_test-1
            test_indices_in_test_data = list(range(n_test))
        
        print(f"进行spots填补，测试spots数量: {n_test}")
        
        # 从test_data中获取测试坐标
        test_coords = test_data.pos
        imputed_expr = self.infer_imputation_spots(test_coords)
        
        # 2. 准备原始数据和填补数据
        original_expr = test_data.y
        
        # 3. 使用简洁的指标计算函数
        try:
            from .metrics import compute_spots_imputation_metrics, print_metrics
        except ImportError:
            # 如果metrics模块不存在，使用utils_analys中的函数
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
            # 使用简洁的指标计算
            # 注意：test_data只包含测试spots，所以使用test_indices_in_test_data
            metrics = compute_spots_imputation_metrics(
                original_data=test_data,
                imputed_data=imputed_expr,
                test_indices=test_indices_in_test_data,
                
            )
            
            # 打印指标
            print_metrics(metrics, title=f"Spots填补指标 - {experiment_name}")
            
            # 保存指标到文件
            if output_dir:
                import os
                import json
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')
                with open(json_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"指标已保存到: {json_path}")
        
        return metrics
    
    def evaluate_spots_imputation_from_perturb(self, original_data, perturb_result,
                                              output_dir='./results', experiment_name='spots_imputation'):
        """
        直接从spots_perturb的输出元组评估spots填补性能
        
        参数:
            original_data: 原始Data对象，包含所有spots
            perturb_result: spots_perturb返回的元组 (train_data, test_data, train_indices, test_indices)
            output_dir: 结果保存目录
            experiment_name: 实验名称
        
        返回:
            metrics_dict: 填补指标字典
        """
        # 解包perturb_result
        train_data, test_data, train_indices, test_indices = perturb_result
        
        # 调用完整的评估函数
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
        评估genes填补性能的简化函数
        
        参数:
            original_data: 原始Data对象，包含完整基因表达
            mask_pattern: 基因掩码模式，布尔矩阵
            output_dir: 结果保存目录
            experiment_name: 实验名称
        
        返回:
            metrics_dict: 填补指标字典
        """
        # 1. 创建masked数据
        import copy
        import torch
        
        masked_data = copy.deepcopy(original_data)
        
        # 应用mask
        if isinstance(mask_pattern, torch.Tensor):
            mask_tensor = mask_pattern
        else:
            mask_tensor = torch.tensor(mask_pattern, dtype=torch.bool)
        
        # 将masked位置的基因表达设为0
        masked_data.x = masked_data.x.clone()
        masked_data.x[mask_tensor] = 0
        
        # 2. 使用infer_imputation_genes进行填补
        print(f"进行genes填补，masked元素数量: {mask_tensor.sum().item()}")
        imputed_expr = self.infer_imputation_genes(masked_data)
        
        # 3. 直接计算指标
        from .utils_analys import compute_imputation_metrics
        
        # 创建填补后的数据对象
        import torch_geometric.data as data
        
        imputed_data = data.Data(
            x=imputed_expr,
            pos=original_data.pos.clone() if hasattr(original_data, 'pos') else None,
            edge_index=original_data.edge_index.clone() if hasattr(original_data, 'edge_index') else None
        )
        
        # 4. 计算指标
        metrics = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_data,
            mask_pattern=mask_pattern,
            output_dir=output_dir,
            experiment_name=experiment_name,
            compute_full_data=False  # 只计算被mask位置的指标
        )
        
        return metrics