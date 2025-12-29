#!/usr/bin/env python3
"""
测试新的简洁指标计算函数
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
from torch_geometric.data import Data

def test_new_metrics():
    print("=" * 60)
    print("测试新的简洁指标计算函数")
    print("=" * 60)
    
    # 测试1: 基本指标计算
    print("\n1. 测试基本指标计算...")
    try:
        from CPS.metrics import compute_simple_metrics, print_metrics
        
        # 创建测试数据
        n_samples = 100
        n_features = 10
        
        np.random.seed(42)
        original = np.random.randn(n_samples, n_features)
        noise = np.random.randn(n_samples, n_features) * 0.1
        imputed = original + noise
        
        # 计算指标
        metrics = compute_simple_metrics(original, imputed)
        
        print(f"✅ 基本指标计算成功")
        print(f"   获取的指标数量: {len(metrics)}")
        print_metrics(metrics, title="基本指标测试")
        
    except Exception as e:
        print(f"❌ 基本指标测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: Spots填补指标
    print("\n2. 测试Spots填补指标...")
    try:
        from CPS.metrics import compute_spots_imputation_metrics
        
        n_spots = 50
        n_genes = 20
        n_test = 10
        
        original_features = np.random.randn(n_spots, n_genes)
        imputed_features = original_features + np.random.randn(n_spots, n_genes) * 0.1
        
        # 创建Data对象
        original_data = Data(
            x=torch.tensor(original_features, dtype=torch.float),
            pos=torch.randn(n_spots, 2)
        )
        
        # 测试索引
        test_indices = np.random.choice(n_spots, size=n_test, replace=False)
        
        # 计算spots填补指标
        metrics = compute_spots_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_features,
            test_indices=test_indices
        )
        
        print(f"✅ Spots填补指标计算成功")
        print(f"   测试spots数量: {n_test}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   R²: {metrics['r2']:.6f}")
        
    except Exception as e:
        print(f"❌ Spots填补指标测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: Genes填补指标
    print("\n3. 测试Genes填补指标...")
    try:
        from CPS.metrics import compute_genes_imputation_metrics
        
        n_spots = 30
        n_genes = 15
        
        original_features = np.random.randn(n_spots, n_genes)
        imputed_features = original_features + np.random.randn(n_spots, n_genes) * 0.1
        
        # 创建mask模式
        mask_ratio = 0.3
        mask_pattern = np.random.rand(n_spots, n_genes) < mask_ratio
        
        # 创建Data对象
        original_data = Data(
            x=torch.tensor(original_features, dtype=torch.float)
        )
        
        # 计算genes填补指标
        metrics = compute_genes_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_features,
            mask_pattern=mask_pattern
        )
        
        print(f"✅ Genes填补指标计算成功")
        print(f"   masked元素数量: {mask_pattern.sum()}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   Pearson相关系数: {metrics['pearson_corr']:.6f}")
        
    except Exception as e:
        print(f"❌ Genes填补指标测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_new_metrics()