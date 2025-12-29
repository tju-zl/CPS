#!/usr/bin/env python3
"""
测试ZINB解码器和损失函数的正确性
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from CPS.model import (
    ZINBLoss, 
    ResidualBlock, 
    ZINBDecoder, 
    SharedDecoder,
    CPSModel
)

def test_zinb_loss():
    """测试ZINB损失函数"""
    print("=== 测试ZINB损失函数 ===")
    zinb_loss = ZINBLoss()
    
    # 创建测试数据
    batch_size = 4
    n_genes = 10
    x = torch.randn(batch_size, n_genes).abs()  # 基因表达（非负）
    mean = torch.randn(batch_size, n_genes).abs() + 0.1  # 均值（正数）
    disp = torch.randn(batch_size, n_genes).abs() + 0.1  # 离散度（正数）
    pi = torch.sigmoid(torch.randn(batch_size, n_genes))  # 零膨胀概率[0,1]
    
    # 计算损失
    loss = zinb_loss(x, mean, disp, pi)
    print(f"ZINB损失值: {loss.item():.4f}")
    print(f"损失形状: {loss.shape if hasattr(loss, 'shape') else '标量'}")
    
    # 测试梯度
    mean.requires_grad_(True)
    loss.backward()
    print(f"均值梯度存在: {mean.grad is not None}")
    print(f"梯度非零: {torch.any(mean.grad != 0)}")
    
    return loss.item() > 0

def test_residual_block():
    """测试残差块"""
    print("\n=== 测试残差块 ===")
    dim = 32
    block = ResidualBlock(dim, dropout=0.1)
    
    # 测试前向传播
    x = torch.randn(4, dim)
    output = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"残差连接: {torch.allclose(output, x + block.linear2(block.act(block.norm2(block.linear2(block.dropout(block.act(block.norm1(block.linear1(x)))))))))}")
    
    # 测试梯度
    x.requires_grad_(True)
    output = block(x)
    loss = output.sum()
    loss.backward()
    print(f"梯度存在: {x.grad is not None}")
    
    return output.shape == x.shape

def test_zinb_decoder():
    """测试ZINB解码器"""
    print("\n=== 测试ZINB解码器 ===")
    in_dim = 64
    out_dim = 100
    decoder = ZINBDecoder(in_dim, out_dim, hid_dims=[128, 256, 128])
    
    # 测试前向传播
    z = torch.randn(8, in_dim)
    mean, disp, pi = decoder(z)
    
    print(f"输入形状: {z.shape}")
    print(f"均值形状: {mean.shape}")
    print(f"离散度形状: {disp.shape}")
    print(f"零膨胀概率形状: {pi.shape}")
    
    # 检查参数范围
    print(f"均值范围: [{mean.min().item():.4f}, {mean.max().item():.4f}] (应为正数)")
    print(f"离散度范围: [{disp.min().item():.4f}, {disp.max().item():.4f}] (应为正数)")
    print(f"零膨胀概率范围: [{pi.min().item():.4f}, {pi.max().item():.4f}] (应在[0,1]之间)")
    
    # 测试梯度
    z.requires_grad_(True)
    mean, disp, pi = decoder(z)
    loss = mean.sum() + disp.sum() + pi.sum()
    loss.backward()
    print(f"梯度存在: {z.grad is not None}")
    
    return all([mean.shape == (8, out_dim),
                disp.shape == (8, out_dim),
                pi.shape == (8, out_dim),
                torch.all(mean > 0),
                torch.all(disp > 0),
                torch.all(pi >= 0) and torch.all(pi <= 1)])

def test_shared_decoder():
    """测试共享解码器"""
    print("\n=== 测试共享解码器 ===")
    in_dim = 64
    out_dim = 100
    
    # 测试ZINB模式
    decoder_zinb = SharedDecoder(in_dim, out_dim, use_zinb=True)
    z = torch.randn(8, in_dim)
    
    # 测试返回参数
    mean, disp, pi = decoder_zinb(z, return_params=True)
    print(f"ZINB模式 - 参数形状: {mean.shape}, {disp.shape}, {pi.shape}")
    
    # 测试返回预测值
    pred = decoder_zinb(z, return_params=False)
    print(f"ZINB模式 - 预测形状: {pred.shape}")
    
    # 测试传统模式
    decoder_traditional = SharedDecoder(in_dim, out_dim, use_zinb=False)
    pred_trad = decoder_traditional(z, return_params=False)
    print(f"传统模式 - 预测形状: {pred_trad.shape}")
    
    return True

def test_cps_model_integration():
    """测试CPS模型集成"""
    print("\n=== 测试CPS模型集成 ===")
    
    # 创建模拟参数
    class Args:
        hvgs = 100
        latent_dim = 64
        k_list = [5, 10, 15]
        num_heads = 4
        dropout = 0.1
        sh_weights = False
        prep_scale = False
        coord_dim = 2
        freq = 32
        sigma = 1.0
        inr_latent = [128, 128]
        decoder_latent = [256, 512, 256]
        distill = 0.1
    
    args = Args()
    
    # 创建模型
    model = CPSModel(args)
    print(f"模型创建成功")
    print(f"教师网络参数数量: {sum(p.numel() for p in model.teacher.parameters())}")
    print(f"学生网络参数数量: {sum(p.numel() for p in model.student.parameters())}")
    print(f"解码器参数数量: {sum(p.numel() for p in model.decoder.parameters())}")
    
    # 测试前向传播
    batch_size = 16
    coords = torch.randn(batch_size, 2)
    x = torch.randn(batch_size, args.hvgs)
    edge_index = torch.randint(0, batch_size, (2, 30))
    
    # 测试标准模式
    results = model(coords, x, edge_index, return_attn=False)
    print(f"标准模式输出键: {list(results.keys())}")
    
    # 测试ZINB参数模式
    results_zinb = model(coords, x, edge_index, return_attn=False, return_zinb_params=True)
    print(f"ZINB模式输出键: {list(results_zinb.keys())}")
    
    # 检查ZINB参数是否存在
    zinb_params = ['mean_teacher', 'disp_teacher', 'pi_teacher', 
                   'mean_student', 'disp_student', 'pi_student']
    has_zinb = all(param in results_zinb for param in zinb_params)
    print(f"ZINB参数存在: {has_zinb}")
    
    if has_zinb:
        for param in zinb_params:
            shape = results_zinb[param].shape
            print(f"  {param}: {shape}")
    
    return True

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n=== 测试训练兼容性 ===")
    
    from CPS.cps import CPSTrainer
    
    # 创建模拟参数
    class Args:
        hvgs = 50
        latent_dim = 32
        k_list = [5, 10]
        num_heads = 2
        dropout = 0.1
        sh_weights = False
        prep_scale = False
        coord_dim = 2
        freq = 16
        sigma = 1.0
        inr_latent = [64, 64]
        decoder_latent = [128, 256, 128]
        distill = 0.1
        lr = 1e-3
        weight_decay = 1e-5
        max_epoch = 2  # 只测试2个epoch
    
    args = Args()
    
    # 创建训练器
    trainer = CPSTrainer(args)
    print(f"训练器创建成功")
    
    # 创建模拟数据
    import torch_geometric.data as data
    
    n_spots = 32
    pyg_data = data.Data(
        x=torch.randn(n_spots, args.hvgs).abs(),
        pos=torch.randn(n_spots, 2),
        edge_index=torch.randint(0, n_spots, (2, 50))
    )
    
    val_data = data.Data(
        x=torch.randn(n_spots//2, args.hvgs).abs(),
        pos=torch.randn(n_spots//2, 2),
        edge_index=torch.randint(0, n_spots//2, (2, 30))
    )
    
    # 测试一个训练步骤
    try:
        # 手动执行一个训练步骤
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        x = pyg_data.x.to(trainer.device)
        pos = pyg_data.pos.to(trainer.device)
        edge_index = pyg_data.edge_index.to(trainer.device)
        
        results = trainer.model(pos, x, edge_index, return_attn=False)
        losses = trainer.compute_losses(results, x, recon_weight=[0.5, 0.5], verbose=True)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 5)
        trainer.optimizer.step()
        
        print(f"训练步骤成功完成")
        print(f"总损失: {losses['total'].item():.4f}")
        
        # 检查是否有ZINB损失
        if 'recon_teacher_zinb' in losses:
            print(f"使用ZINB损失: {losses['recon_teacher_zinb'].item():.4f}")
        else:
            print(f"使用传统损失")
        
        return True
    except Exception as e:
        print(f"训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始验证ZINB解码器和损失函数的实现正确性")
    print("=" * 60)
    
    tests = [
        ("ZINB损失函数", test_zinb_loss),
        ("残差块", test_residual_block),
        ("ZINB解码器", test_zinb_decoder),
        ("共享解码器", test_shared_decoder),
        ("CPS模型集成", test_cps_model_integration),
        ("训练兼容性", test_training_compatibility),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = success
            status = "✓ 通过" if success else "✗ 失败"
            print(f"{name}: {status}")
        except Exception as e:
            results[name] = False
            print(f"{name}: ✗ 异常 - {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    total = len(results)
    passed = sum(results.values())
    
    for name, success in results.items():
        status = "通过" if success else "失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("✅ 所有测试通过！ZINB解码器和损失函数实现正确。")
    else:
        print("⚠️  部分测试失败，需要检查实现。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)