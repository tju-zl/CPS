#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

print("测试config.py修复...")

from CPS.config import config, str_to_int_list

print("✓ 成功导入模块")

# 测试类型转换函数
test_cases = [
    ("1,2,3", [1, 2, 3]),
    ("256,512", [256, 512]),
    ([1, 2, 3], [1, 2, 3]),  # 已经是列表
]

for input_val, expected in test_cases:
    result = str_to_int_list(input_val)
    print(f"  输入: {input_val} -> 输出: {result}, 期望: {expected}")
    if result != expected:
        print(f"  ✗ 转换失败: {result} != {expected}")
        sys.exit(1)

print("✓ 类型转换函数测试通过")

# 测试参数解析
parser = config()

# 测试空参数列表（使用默认值）
args = parser.parse_args([])
print(f"\n默认参数测试:")
print(f"  k_list: {args.k_list}, 类型: {type(args.k_list)}")
print(f"  inr_latent: {args.inr_latent}, 类型: {type(args.inr_latent)}")
print(f"  decoder_latent: {args.decoder_latent}, 类型: {type(args.decoder_latent)}")

# 检查是否为列表
if not isinstance(args.k_list, list):
    print(f"✗ k_list不是列表: {type(args.k_list)}")
    sys.exit(1)
if not isinstance(args.inr_latent, list):
    print(f"✗ inr_latent不是列表: {type(args.inr_latent)}")
    sys.exit(1)
if not isinstance(args.decoder_latent, list):
    print(f"✗ decoder_latent不是列表: {type(args.decoder_latent)}")
    sys.exit(1)

print("✓ 默认参数类型检查通过")

# 测试命令行参数
test_args = [
    '--k_list', '2,4,6,8',
    '--inr_latent', '128,256,512',
    '--decoder_latent', '512,256'
]

args2 = parser.parse_args(test_args)
print(f"\n命令行参数测试:")
print(f"  k_list: {args2.k_list}, 期望: [2, 4, 6, 8]")
print(f"  inr_latent: {args2.inr_latent}, 期望: [128, 256, 512]")
print(f"  decoder_latent: {args2.decoder_latent}, 期望: [512, 256]")

if args2.k_list != [2, 4, 6, 8]:
    print(f"✗ k_list解析错误: {args2.k_list}")
    sys.exit(1)
if args2.inr_latent != [128, 256, 512]:
    print(f"✗ inr_latent解析错误: {args2.inr_latent}")
    sys.exit(1)
if args2.decoder_latent != [512, 256]:
    print(f"✗ decoder_latent解析错误: {args2.decoder_latent}")
    sys.exit(1)

print("✓ 命令行参数解析测试通过")

print("\n" + "="*60)
print("所有测试通过! config.py修复成功。")
print("="*60)