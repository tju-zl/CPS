#!/usr/bin/env python3
"""
最简单的测试 - 只检查导入和基本功能
"""

import sys
import os
sys.path.append('.')

print("Python版本:", sys.version)
print("当前目录:", os.getcwd())

# 首先检查基本导入
try:
    import numpy as np
    print("✅ numpy 导入成功")
except ImportError as e:
    print(f"❌ numpy 导入失败: {e}")

try:
    import pandas as pd
    print("✅ pandas 导入成功")
except ImportError as e:
    print(f"❌ pandas 导入失败: {e}")

# 尝试导入utils_analys
print("\n尝试导入 CPS.utils_analys...")
try:
    from CPS.utils_analys import compute_imputation_metrics
    print("✅ compute_imputation_metrics 导入成功")
    
    # 创建最简单的测试数据
    import numpy as np
    
    # 使用numpy数组而不是Data对象
    n_spots = 10
    n_genes = 5
    
    original_data = np.random.randn(n_spots, n_genes)
    imputed_data = original_data + np.random.randn(n_spots, n_genes) * 0.1
    
    print("\n测试通用指标计算...")
    try:
        metrics = compute_imputation_metrics(
            original_data=original_data,
            imputed_data=imputed_data,
            experiment_name='simple_test',
            output_dir='./simple_test_results'
        )
        print(f"✅ 指标计算成功")
        print(f"   获取的指标数量: {len(metrics)}")
        
        # 打印一些关键指标
        for key in ['mse', 'rmse', 'mae', 'r2_score']:
            if key in metrics:
                print(f"   {key}: {metrics[key]:.6f}")
                
    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

# 清理
import shutil
if os.path.exists('./simple_test_results'):
    shutil.rmtree('./simple_test_results')
    print("\n清理测试文件完成")

print("\n测试结束")