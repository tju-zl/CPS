#!/usr/bin/env python3
"""
简单测试修复后的compute_imputation_metrics函数
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import torch
from torch_geometric.data import Data

def test_fixed_function():
    """测试修复后的函数"""
    print("=" * 60)
    print("测试修复后的 compute_imputation_metrics 函数")
    print("=" * 60)
    
    try:
        from CPS.utils_analys import compute_imputation_metrics
        print("✅ 成功导入函数")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模拟数据
    n_spots = 30
    n_genes = 15
    
    # 原始数据
    np.random.seed(42)
    original_features = np.random.randn(n_spots, n_genes)
    
    # 填补数据（添加一些噪声）
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
    
    # 测试1: Spots填补指标
    print("\n1. 测试Spots填补指标...")
    test_indices = np.random.choice(n_spots, size=8, replace=False)
    
    try:
        metrics_spots = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_data,
            test_indices=test_indices,
            experiment_name='test_spots_fix',
            output_dir='./test_fix_results'
        )
        
        print(f"✅ Spots填补测试成功")
        print(f"   获取的指标数量: {len(metrics_spots)}")
        
        # 检查关键指标是否存在
        required_keys = ['test_spots_mse', 'test_spots_rmse', 'test_spots_mae', 'test_spots_r2']
        for key in required_keys:
            if key in metrics_spots:
                print(f"   {key}: {metrics_spots[key]:.6f}")
            else:
                print(f"   ⚠️ 缺少指标: {key}")
                
    except Exception as e:
        print(f"❌ Spots填补测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: Genes填补指标
    print("\n2. 测试Genes填补指标...")
    mask_ratio = 0.3
    mask_pattern = np.zeros((n_spots, n_genes), dtype=bool)
    n_mask_per_spot = int(n_genes * mask_ratio)
    
    for i in range(n_spots):
        gene_indices = np.random.choice(n_genes, size=n_mask_per_spot, replace=False)
        mask_pattern[i, gene_indices] = True
    
    try:
        metrics_genes = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_data,
            mask_pattern=mask_pattern,
            experiment_name='test_genes_fix',
            output_dir='./test_fix_results'
        )
        
        print(f"✅ Genes填补测试成功")
        print(f"   获取的指标数量: {len(metrics_genes)}")
        
        # 检查关键指标是否存在
        required_keys = ['masked_genes_mse', 'masked_genes_rmse', 'masked_genes_mae', 'masked_genes_r2']
        for key in required_keys:
            if key in metrics_genes:
                print(f"   {key}: {metrics_genes[key]:.6f}")
            else:
                print(f"   ⚠️ 缺少指标: {key}")
                
        # 检查通用指标键
        if 'mse' in metrics_genes:
            print(f"   通用MSE键: {metrics_genes['mse']:.6f}")
        if 'rmse' in metrics_genes:
            print(f"   通用RMSE键: {metrics_genes['rmse']:.6f}")
            
    except Exception as e:
        print(f"❌ Genes填补测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理测试文件
    import shutil
    if os.path.exists('./test_fix_results'):
        shutil.rmtree('./test_fix_results')
        print("\n清理测试文件完成")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_fixed_function()