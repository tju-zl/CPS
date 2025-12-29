#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

try:
    from CPS.config import config
    print("✓ 成功导入config模块")
    
    parser = config()
    print("✓ 成功创建parser")
    
    args = parser.parse_args([])
    print("✓ 成功解析参数")
    
    print(f"默认参数:")
    print(f"  hvgs: {args.hvgs}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  max_epoch: {args.max_epoch}")
    
    # 修改参数
    args.hvgs = 100
    args.latent_dim = 64
    args.max_epoch = 10
    
    print(f"\n修改后参数:")
    print(f"  hvgs: {args.hvgs}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  max_epoch: {args.max_epoch}")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()