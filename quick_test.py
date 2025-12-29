import sys
sys.path.append('.')

# 测试1: 直接导入
print("测试1: 直接导入config模块")
from CPS import config
print("✓ 成功导入CPS.config")

# 测试2: 调用config函数
print("\n测试2: 调用config()函数")
parser = config.config()
print("✓ 成功创建parser")

# 测试空参数
args = parser.parse_args([])
print("✓ 成功解析空参数")

# 检查关键参数
print(f"  k_list: {args.k_list}")
print(f"  inr_latent: {args.inr_latent}")
print(f"  decoder_latent: {args.decoder_latent}")

# 检查类型
print(f"  k_list类型: {type(args.k_list)}")
print(f"  inr_latent类型: {type(args.inr_latent)}")

# 测试3: 测试类型转换函数
print("\n测试3: 测试str_to_int_list函数")
result = config.str_to_int_list("1,2,3")
print(f"  '1,2,3' -> {result}")

result2 = config.str_to_int_list([4,5,6])
print(f"  [4,5,6] -> {result2}")

print("✓ 类型转换函数测试通过")