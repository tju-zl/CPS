#!/usr/bin/env python3
"""
测试CPS/utils_analys.py中的compute_imputation_metrics函数
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import torch
from torch_geometric.data import Data

def test_compute_imputation_metrics():
    """测试compute_imputation_metrics函数"""
    print("=" * 60)
    print("测试 compute_imputation_metrics 函数")
    print("=" * 60)
    
    # 导入函数
    from CPS.utils_analys import compute_imputation_metrics
    
    # 创建模拟数据
    n_spots = 100
    n_genes = 50
    
    # 原始数据
    original_features = np.random.randn(n_spots, n_genes)
    
    # 填补数据（添加一些噪声）
    np.random.seed(42)
    noise = np.random.randn(n_spots, n_genes) * 0.1
    imputed_features = original_features + noise
    
    # 创建Data对象
    original_data = Data(
        x=torch.tensor(original_features, dtype=torch.float),
        pos=torch.randn(n_spots, 2)
    )
    
    imputed_data = Data(
        x=torch.tensor(imputed_features, dtype=torch.float),
        pos=original_data.pos
    )
    
    # 测试1: 通用指标计算
    print("\n1. 测试通用指标计算...")
    metrics = compute_imputation_metrics(
        original_data=original_data,
        imputed_data=imputed_data,
        experiment_name='test_general_metrics',
        output_dir='./test_results'
    )
    
    # 验证关键指标存在
    required_metrics = ['mse', 'rmse', 'mae', 'r2_score', 'pearson_correlation']
    for metric in required_metrics:
        assert metric in metrics, f"缺少指标: {metric}"
        print(f"  {metric}: {metrics[metric]:.6f}")
    
    # 测试2: Spots填补指标
    print("\n2. 测试Spots填补指标...")
    test_indices = np.random.choice(n_spots, size=20, replace=False)
    
    metrics_spots = compute_imputation_metrics(
        original_data=original_data,
        imputed_data=imputed_data,
        test_indices=test_indices,
        experiment_name='test_spots_imputation',
        output_dir='./test_results'
    )
    
    # 验证spots特定指标
    spots_metrics = ['test_spots_mse', 'test_spots_rmse', 'test_spots_mae', 'test_spots_r2']
    for metric in spots_metrics:
        if metric in metrics_spots:
            print(f"  {metric}: {metrics_spots[metric]:.6f}")
    
    # 测试3: Genes填补指标
    print("\n3. 测试Genes填补指标...")
    mask_ratio = 0.3
    mask_pattern = np.zeros((n_spots, n_genes), dtype=bool)
    n_mask_per_spot = int(n_genes * mask_ratio)
    
    for i in range(n_spots):
        gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
        mask_pattern[i, gene_indices] = True
    
    metrics_genes = compute_imputation_metrics(
        original_data=original_data,
        imputed_data=imputed_data,
        mask_pattern=mask_pattern,
        experiment_name='test_genes_imputation',
        output_dir='./test_results'
    )
    
    # 验证genes特定指标
    genes_metrics = ['masked_genes_mse', 'masked_genes_rmse', 'masked_genes_mae', 'masked_genes_r2']
    for metric in genes_metrics:
        if metric in metrics_genes:
            print(f"  {metric}: {metrics_genes[metric]:.6f}")
    
    # 验证基因级别指标
    if 'per_gene_metrics' in metrics_genes:
        print(f"  基因级别指标数量: {len(metrics_genes['per_gene_metrics'])}")
        print(f"  基因平均MSE: {metrics_genes.get('gene_mse_mean', 'N/A'):.6f}")
        print(f"  基因平均相关系数: {metrics_genes.get('gene_correlation_mean', 'N/A'):.6f}")
    
    # 检查文件是否保存
    import glob
    json_files = glob.glob('./test_results/*.json')
    csv_files = glob.glob('./test_results/*.csv')
    
    print(f"\n4. 检查保存的文件:")
    print(f"  JSON文件: {len(json_files)}个")
    print(f"  CSV文件: {len(csv_files)}个")
    
    for file in json_files[:2]:  # 显示前2个文件
        print(f"    - {os.path.basename(file)}")
    
    # 清理测试文件
    import shutil
    if os.path.exists('./test_results'):
        shutil.rmtree('./test_results')
        print("\n清理测试文件完成")
    
    print("\n✅ compute_imputation_metrics 测试通过!")

def test_integration_with_perturb_methods():
    """测试与perturb方法的集成"""
    print("\n" + "=" * 60)
    print("测试与perturb方法的集成")
    print("=" * 60)
    
    try:
        from CPS.utils_adata import SpatialGraphBuilder
        
        # 创建模拟配置
        class Args:
            def __init__(self):
                self.max_neighbors = 6
                self.radius = 50
                self.self_loops = True
                self.flow = 'source_to_target'
        
        # 创建模拟数据
        class MockAnnData:
            def __init__(self, n_obs, n_vars):
                self.n_obs = n_obs
                self.n_vars = n_vars
                self.obsm = {}
                self.obs = {}
                self.var = {}
                self.X = np.random.rand(n_obs, n_vars)
        
        n_spots = 50
        n_genes = 30
        
        adata = MockAnnData(n_spots, n_genes * 2)
        adata.obsm['spatial'] = np.random.rand(n_spots, 2) * 100
        adata.obsm['hvg_features'] = np.random.randn(n_spots, n_genes)
        
        args = Args()
        builder = SpatialGraphBuilder(args)
        
        # 测试spots_perturb
        print("1. 测试spots_perturb...")
        train_data, test_data, train_idx, test_idx = builder.spots_perturb(
            adata, mask_ratio=0.2, seed=42
        )
        
        print(f"  训练数据: {train_data}")
        print(f"  测试数据: {test_data}")
        print(f"  训练索引: {len(train_idx)}个")
        print(f"  测试索引: {len(test_idx)}个")
        
        # 测试genes_perturb
        print("\n2. 测试genes_perturb...")
        train_data2, test_data2, mask_pattern = builder.genes_perturb(
            adata, mask_ratio=0.3, seed=42, train_mask_only=True
        )
        
        print(f"  训练数据: {train_data2}")
        print(f"  测试数据: {test_data2}")
        print(f"  掩码模式形状: {mask_pattern.shape}")
        
        print("\n✅ 与perturb方法集成测试通过!")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_compute_imputation_metrics()
        test_integration_with_perturb_methods()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)