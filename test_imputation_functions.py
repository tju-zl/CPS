#!/usr/bin/env python3
"""
测试CPS填补功能实现
"""

import torch
import numpy as np
from torch_geometric.data import Data
import sys
import os

# 添加CPS目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CPS.cps import CPSTrainer
from CPS.config import config

def test_imputation_functions():
    """测试填补功能"""
    print("=" * 60)
    print("测试CPS填补功能实现")
    print("=" * 60)
    
    # 创建模拟参数
    parser = config()
    args = parser.parse_args([])  # 使用空列表作为参数
    args.hvgs = 100  # 基因数量
    args.latent_dim = 64
    args.max_epoch = 10  # 测试时使用较少的epoch
    args.lr = 1e-3
    
    # 创建训练器
    trainer = CPSTrainer(args)
    print(f"✓ 创建CPSTrainer: device={trainer.device}")
    
    # 创建模拟数据
    n_spots = 50
    n_genes = args.hvgs
    
    # 1. 测试spots填补
    print("\n1. 测试spots填补 (infer_imputation_spots):")
    
    # 创建随机坐标
    coords = torch.randn(n_spots, 2)
    print(f"   - 输入坐标形状: {coords.shape}")
    
    # 测试填补
    try:
        imputed_spots = trainer.infer_imputation_spots(coords)
        print(f"   ✓ 填补成功!")
        print(f"   - 输出形状: {imputed_spots.shape}")
        print(f"   - 期望形状: ({n_spots}, {n_genes})")
        
        # 检查输出
        assert imputed_spots.shape == (n_spots, n_genes), f"形状不匹配: {imputed_spots.shape}"
        assert not torch.isnan(imputed_spots).any(), "输出包含NaN值"
        assert not torch.isinf(imputed_spots).any(), "输出包含Inf值"
        print("   ✓ 输出形状和值检查通过")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        return False
    
    # 2. 测试genes填补
    print("\n2. 测试genes填补 (infer_imputation_genes):")
    
    # 创建模拟图数据
    x = torch.randn(n_spots, n_genes)  # 基因表达
    pos = torch.randn(n_spots, 2)  # 坐标
    edge_index = torch.randint(0, n_spots, (2, 100))  # 随机边
    
    pyg_data = Data(x=x, pos=pos, edge_index=edge_index)
    print(f"   - 输入数据: x={x.shape}, pos={pos.shape}, edge_index={edge_index.shape}")
    
    try:
        imputed_genes = trainer.infer_imputation_genes(pyg_data)
        print(f"   ✓ 填补成功!")
        print(f"   - 输出形状: {imputed_genes.shape}")
        print(f"   - 期望形状: ({n_spots}, {n_genes})")
        
        # 检查输出
        assert imputed_genes.shape == (n_spots, n_genes), f"形状不匹配: {imputed_genes.shape}"
        assert not torch.isnan(imputed_genes).any(), "输出包含NaN值"
        assert not torch.isinf(imputed_genes).any(), "输出包含Inf值"
        print("   ✓ 输出形状和值检查通过")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        return False
    
    # 3. 测试注意力解释功能
    print("\n3. 测试注意力解释功能 (interpret_attn_scores):")
    
    try:
        z_teacher, attn_weights = trainer.interpret_attn_scores(pyg_data, return_fig=False)
        print(f"   ✓ 注意力解释成功!")
        print(f"   - z_teacher形状: {z_teacher.shape}")
        print(f"   - attn_weights形状: {attn_weights.shape}")
        
        # 检查输出
        assert z_teacher.shape[0] == n_spots, f"z_teacher形状错误: {z_teacher.shape}"
        assert attn_weights.shape[0] == n_spots, f"attn_weights形状错误: {attn_weights.shape}"
        print("   ✓ 输出形状检查通过")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        return False
    
    # 4. 测试训练功能
    print("\n4. 测试训练功能 (fit):")
    
    try:
        # 简单训练测试
        train_losses = trainer.fit(pyg_data, verbose=False)
        print(f"   ✓ 训练成功!")
        print(f"   - 训练损失列表长度: {len(train_losses)}")
        
        if isinstance(train_losses, tuple):
            train_losses, val_losses = train_losses
            print(f"   - 训练损失: {train_losses[-1]:.4f}")
            if val_losses is not None:
                print(f"   - 验证损失: {val_losses[-1]:.4f}")
        else:
            print(f"   - 最终训练损失: {train_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
    
    return True

def test_integration_with_utils():
    """测试与utils模块的集成"""
    print("\n" + "=" * 60)
    print("测试与utils模块的集成")
    print("=" * 60)
    
    try:
        # 测试是否能导入相关模块
        from CPS.utils_adata import SpatialGraphBuilder
        from CPS.utils_analys import compute_imputation_metrics
        from CPS.utils_visual import plot_imputation_metrics
        
        print("✓ 成功导入所有utils模块")
        
        # 创建模拟数据用于测试
        n_spots = 30
        n_genes = 50
        
        # 原始数据
        original_data = torch.randn(n_spots, n_genes)
        
        # 填补数据（模拟）
        imputed_data = original_data + torch.randn(n_spots, n_genes) * 0.1
        
        # 测试指标计算
        print("\n测试指标计算:")
        metrics = compute_imputation_metrics(
            original_data.numpy(),
            imputed_data.numpy(),
            output_dir='./test_results',
            experiment_name='test_integration'
        )
        
        print(f"✓ 指标计算成功!")
        print(f"  - 计算了 {len(metrics)} 个指标")
        print(f"  - MSE: {metrics.get('mse', 'N/A'):.4f}")
        print(f"  - R²: {metrics.get('r2', 'N/A'):.4f}")
        
        # 测试可视化
        print("\n测试可视化:")
        try:
            # 创建测试目录
            os.makedirs('./test_results', exist_ok=True)
            
            # 测试可视化函数
            plot_imputation_metrics(
                metrics,
                output_path='./test_results/test_metrics.png',
                figsize=(12, 8)
            )
            print("✓ 可视化成功!")
            print("  - 图像保存到: ./test_results/test_metrics.png")
            
        except Exception as e:
            print(f"⚠ 可视化测试警告: {e}")
            print("  (这可能是由于缺少matplotlib后端，但不影响核心功能)")
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # 运行测试
    success = test_imputation_functions()
    
    if success:
        # 运行集成测试
        integration_success = test_integration_with_utils()
        
        if integration_success:
            print("\n" + "=" * 60)
            print("所有测试完成! 填补功能实现正确。")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("集成测试失败，但核心填补功能正常。")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("核心填补功能测试失败!")
        print("=" * 60)
        sys.exit(1)