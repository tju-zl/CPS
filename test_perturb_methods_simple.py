#!/usr/bin/env python3
"""
测试CPS/utils_adata.py中的spots_perturb和genes_perturb方法
简化版本，不依赖scanpy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import torch

# 创建一个简单的配置类
class Args:
    def __init__(self):
        self.max_neighbors = 6
        self.radius = 50
        self.self_loops = True
        self.flow = 'source_to_target'

# 创建一个简单的模拟AnnData类
class MockAnnData:
    def __init__(self, n_obs, n_vars):
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.obsm = {}
        self.obs = {}
        self.var = {}
        self.X = np.random.rand(n_obs, n_vars)
        
    def __repr__(self):
        return f"MockAnnData(n_obs={self.n_obs}, n_vars={self.n_vars})"

def create_mock_adata(n_spots=100, n_genes=50):
    """创建模拟的AnnData对象用于测试"""
    adata = MockAnnData(n_spots, n_genes * 2)  # 原始表达矩阵
    
    # 创建随机空间坐标
    spatial_coords = np.random.rand(n_spots, 2) * 100
    
    # 创建随机基因表达数据（HVG特征）
    hvg_features = np.random.randn(n_spots, n_genes)
    
    adata.obsm['spatial'] = spatial_coords
    adata.obsm['hvg_features'] = hvg_features
    
    return adata

def test_spots_perturb():
    """测试spots_perturb方法"""
    print("=" * 60)
    print("测试 spots_perturb 方法")
    print("=" * 60)
    
    # 创建模拟数据
    adata = create_mock_adata(n_spots=100, n_genes=50)
    args = Args()
    
    # 导入SpatialGraphBuilder
    from CPS.utils_adata import SpatialGraphBuilder
    builder = SpatialGraphBuilder(args)
    
    # 测试mask 20%的spots
    mask_ratio = 0.2
    train_data, test_data, train_indices, test_indices = builder.spots_perturb(
        adata, mask_ratio, method='rknn', seed=42
    )
    
    # 验证结果
    print(f"总spots数: {adata.n_obs}")
    print(f"训练spots数: {len(train_indices)}")
    print(f"测试spots数: {len(test_indices)}")
    print(f"Mask比例: {mask_ratio}, 预期测试spots: {int(adata.n_obs * mask_ratio)}")
    
    # 验证训练数据
    print(f"\n训练数据验证:")
    print(f"  - 特征形状: {train_data.x.shape}")
    print(f"  - 位置形状: {train_data.pos.shape}")
    print(f"  - 边索引形状: {train_data.edge_index.shape if train_data.edge_index is not None else 'None'}")
    print(f"  - 节点数: {train_data.num_nodes}")
    
    # 验证测试数据
    print(f"\n测试数据验证:")
    print(f"  - 特征形状: {test_data.x.shape}")
    print(f"  - 位置形状: {test_data.pos.shape}")
    print(f"  - 边索引: {test_data.edge_index} (应为None)")
    print(f"  - 节点数: {test_data.num_nodes}")
    
    # 验证索引不重叠
    train_set = set(train_indices)
    test_set = set(test_indices)
    intersection = train_set.intersection(test_set)
    print(f"\n索引验证:")
    print(f"  - 训练和测试索引交集: {len(intersection)} (应为0)")
    print(f"  - 总唯一索引数: {len(train_set) + len(test_set)} (应为{adata.n_obs})")
    
    assert len(train_indices) + len(test_indices) == adata.n_obs, "索引总数不正确"
    assert len(intersection) == 0, "训练和测试索引有重叠"
    assert test_data.edge_index is None, "测试数据不应包含图结构"
    assert train_data.edge_index is not None, "训练数据应包含图结构"
    
    print("\n✅ spots_perturb 测试通过!")

def test_genes_perturb():
    """测试genes_perturb方法"""
    print("\n" + "=" * 60)
    print("测试 genes_perturb 方法")
    print("=" * 60)
    
    # 创建模拟数据
    adata = create_mock_adata(n_spots=50, n_genes=30)
    args = Args()
    
    # 导入SpatialGraphBuilder
    from CPS.utils_adata import SpatialGraphBuilder
    builder = SpatialGraphBuilder(args)
    
    # 测试mask 30%的基因
    mask_ratio = 0.3
    train_data, test_data, mask_pattern = builder.genes_perturb(
        adata, mask_ratio, method='rknn', seed=42, train_mask_only=True
    )
    
    # 验证结果
    print(f"总spots数: {adata.n_obs}")
    print(f"总基因数: {adata.obsm['hvg_features'].shape[1]}")
    print(f"Mask比例: {mask_ratio}")
    print(f"掩码模式形状: {mask_pattern.shape}")
    
    # 验证训练数据
    print(f"\n训练数据验证 (train_mask_only=True):")
    print(f"  - 特征形状: {train_data.x.shape}")
    print(f"  - 位置形状: {train_data.pos.shape}")
    print(f"  - 边索引形状: {train_data.edge_index.shape if train_data.edge_index is not None else 'None'}")
    
    # 验证测试数据
    print(f"\n测试数据验证:")
    print(f"  - 特征形状: {test_data.x.shape}")
    print(f"  - 边索引: {test_data.edge_index} (应为None)")
    
    # 验证掩码应用
    n_spots, n_genes = mask_pattern.shape
    expected_masked_per_spot = int(n_genes * mask_ratio)
    
    print(f"\n掩码验证:")
    print(f"  - 每个spot预期mask的基因数: {expected_masked_per_spot}")
    
    # 检查每个spot实际被mask的基因数
    for i in range(min(3, n_spots)):  # 只检查前3个spot
        masked_count = mask_pattern[i].sum().item()
        print(f"  - Spot {i}: {masked_count}个基因被mask")
        assert masked_count == expected_masked_per_spot, f"Spot {i}的mask基因数不正确"
    
    # 验证训练数据中的mask值
    original_features = torch.tensor(adata.obsm['hvg_features'], dtype=torch.float)
    train_features = train_data.x
    test_features = test_data.x
    
    # 检查被mask的位置是否为0
    masked_positions = mask_pattern
    train_masked_values = train_features[masked_positions]
    print(f"\n训练数据中被mask的值:")
    print(f"  - 被mask的位置数: {masked_positions.sum().item()}")
    print(f"  - 被mask的值是否全为0: {torch.all(train_masked_values == 0.0)}")
    
    # 检查测试数据是否完整（当train_mask_only=True时）
    test_original_diff = torch.abs(test_features - original_features).sum()
    print(f"  - 测试数据与原始数据的差异: {test_original_diff.item()} (应为0)")
    
    assert torch.all(train_masked_values == 0.0), "训练数据中被mask的值不是0"
    assert test_original_diff == 0, "测试数据应与原始数据相同"
    
    print("\n✅ genes_perturb 测试通过!")

def test_genes_perturb_both_masked():
    """测试genes_perturb方法，训练和测试都mask"""
    print("\n" + "=" * 60)
    print("测试 genes_perturb (train_mask_only=False)")
    print("=" * 60)
    
    # 创建模拟数据
    adata = create_mock_adata(n_spots=30, n_genes=20)
    args = Args()
    
    # 导入SpatialGraphBuilder
    from CPS.utils_adata import SpatialGraphBuilder
    builder = SpatialGraphBuilder(args)
    
    # 测试mask 40%的基因，训练和测试都mask
    mask_ratio = 0.4
    train_data, test_data, mask_pattern = builder.genes_perturb(
        adata, mask_ratio, method='rknn', seed=42, train_mask_only=False
    )
    
    # 验证结果
    print(f"总spots数: {adata.n_obs}")
    print(f"总基因数: {adata.obsm['hvg_features'].shape[1]}")
    
    # 检查训练和测试数据都有被mask的值
    train_masked_count = (train_data.x == 0.0).sum().item()
    test_masked_count = (test_data.x == 0.0).sum().item()
    
    print(f"\n训练数据中被mask的值数量: {train_masked_count}")
    print(f"测试数据中被mask的值数量: {test_masked_count}")
    print(f"两者都应有被mask的值: {train_masked_count > 0 and test_masked_count > 0}")
    
    assert train_masked_count > 0, "训练数据应有被mask的值"
    assert test_masked_count > 0, "测试数据应有被mask的值"
    
    print("\n✅ genes_perturb (train_mask_only=False) 测试通过!")

if __name__ == "__main__":
    try:
        test_spots_perturb()
        test_genes_perturb()
        test_genes_perturb_both_masked()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)