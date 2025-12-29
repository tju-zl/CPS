# CPS 注意力机制改进计划

## 📋 问题概述

**核心问题**: CPS模型训练200次以上后，注意力得分的多头均值退化成相同，无法看出在不同空间域的差异。

**用户已尝试的修改**:
1. 修改图卷积的alpha参数
2. 修改自环设置
3. 尝试共享QKV权重

**当前效果**: 这些修改有一定帮助，但未根本解决问题。

## 🔍 根本原因分析

### 1. 温度参数问题
- **位置**: `CPS/model.py`第120行
- **问题**: 固定温度参数`temperature=2.3`，`exp(2.3)≈9.97`过大
- **影响**: 导致softmax后的注意力权重过于均匀

### 2. 查询向量设计问题
- **位置**: `CPS/model.py`第138行（当`share_weights=False`时）
- **问题**: 所有尺度的查询向量都来自第一个尺度（`scale_features[:,0,:]`）
- **影响**: 限制了查询多样性，多头容易收敛到相同模式

### 3. 注意力dropout策略问题
- **位置**: `CPS/model.py`第151行
- **问题**: 在softmax后应用dropout
- **影响**: 可能导致注意力信息丢失，训练后期模型学习忽略dropout

### 4. 缺乏多样性约束
- **问题**: 没有明确的机制鼓励多头学习不同模式
- **影响**: 多头注意力容易发生模式坍塌

## 🚀 改进方案

### 阶段一：立即实施的快速修复（1-2天）

#### 1.1 温度参数优化
```python
# 当前代码（model.py第120行）:
self.temperature = nn.Parameter(torch.ones(1) * 2.3)

# 改进方案：
self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # 更合理的初始值
self.temperature_min = 0.1
self.temperature_max = 5.0

# 在forward中（第148行）:
temperature = self.temperature.clamp(self.temperature_min, self.temperature_max)
scale = torch.exp(temperature) / (self.head_dim ** 0.5)
```

#### 1.2 查询向量多样化
```python
# 当前代码（model.py第138行）:
q = q_proj(scale_features[:,0,:])  # 所有尺度使用相同的查询源

# 改进方案1：每个尺度使用自己的特征
q = q_proj(scale_features[:,i,:])  # 第i个尺度使用第i个尺度的特征

# 或改进方案2：混合查询源
if self.share_weights:
    # 使用所有尺度的加权平均
    query_source = scale_features.mean(dim=1)
else:
    # 每个尺度使用自己的特征
    query_source = scale_features[:,i,:]
```

#### 1.3 Dropout策略调整
```python
# 当前代码（model.py第150-151行）:
attn_weights = F.softmax(attn_scores, dim=1)
attn_weights = self.dropout(attn_weights)

# 改进方案：在softmax前应用dropout
attn_scores = self.dropout(attn_scores)  # 先dropout
attn_weights = F.softmax(attn_scores, dim=1)  # 后softmax
```

### 阶段二：中级改进（3-5天）

#### 2.1 注意力多样性正则化
```python
class AttentionDiversityLoss(nn.Module):
    def __init__(self, lambda_div=0.1):
        super().__init__()
        self.lambda_div = lambda_div
    
    def forward(self, attn_weights):
        # attn_weights形状: (N, S, H)
        # 计算头间相似度
        attn_flat = attn_weights.mean(dim=1)  # (N, H)
        similarity = F.cosine_similarity(
            attn_flat.unsqueeze(1),  # (N, 1, H)
            attn_flat.unsqueeze(0),  # (1, N, H)
            dim=2
        )
        # 排除对角线
        mask = 1 - torch.eye(similarity.size(0), device=similarity.device)
        diversity_loss = (similarity * mask).sum() / mask.sum()
        
        return self.lambda_div * diversity_loss

# 在训练循环中添加
diversity_loss = diversity_criterion(attn_weights)
total_loss = losses['total'] + diversity_loss
```

#### 2.2 多头独立温度参数
```python
# 每个注意力头有自己的温度参数
self.temperatures = nn.Parameter(torch.ones(num_heads) * 1.0)

# 在forward中
scale = torch.exp(self.temperatures).view(1, 1, -1) / (self.head_dim ** 0.5)
attn_scores = torch.einsum('nhd,nshd->nsh', query, keys) / (self.head_dim ** 0.5)
attn_scores = attn_scores * scale  # 每个头不同的缩放
```

#### 2.3 尺度特异性增强
```python
def compute_scale_specificity(attn_weights, spatial_coords, radius=50):
    """鼓励不同尺度关注不同的空间区域"""
    specificity_loss = 0
    n_scales = attn_weights.shape[1]
    
    for i in range(n_scales):
        for j in range(i+1, n_scales):
            # 计算两个尺度注意力权重的空间相关性
            attn_i = attn_weights[:, i, :].mean(dim=1)  # (N,)
            attn_j = attn_weights[:, j, :].mean(dim=1)  # (N,)
            
            # 计算局部空间相关性
            local_corr = compute_local_correlation(
                attn_i, attn_j, spatial_coords, radius
            )
            # 惩罚高相关性（鼓励差异）
            specificity_loss += torch.abs(local_corr)
    
    return specificity_loss / (n_scales * (n_scales - 1) / 2)
```

### 阶段三：高级架构改进（1-2周）

#### 3.1 可学习查询向量
```python
class LearnableQueryAttention(nn.Module):
    def __init__(self, num_heads, head_dim, num_scales):
        super().__init__()
        # 可学习的查询向量，每个头、每个尺度独立
        self.learnable_queries = nn.Parameter(
            torch.randn(num_heads, num_scales, head_dim)
        )
        # 可学习的查询权重，决定每个尺度的重要性
        self.query_weights = nn.Parameter(torch.ones(num_heads, num_scales))
    
    def forward(self, scale_features):
        N = scale_features.shape[0]
        # 生成查询向量
        queries = torch.einsum('hsd,hs->hd', 
                              self.learnable_queries,
                              F.softmax(self.query_weights, dim=1))
        queries = queries.unsqueeze(0).expand(N, -1, -1)  # (N, H, D_h)
        return queries
```

#### 3.2 注意力头专业化
```python
class SpecializedAttentionHeads(nn.Module):
    def __init__(self, num_heads, specialization_types=['local', 'global', 'boundary']):
        super().__init__()
        self.specialization_types = specialization_types
        self.num_specializations = len(specialization_types)
        
        # 每个专业化类型有对应的头
        self.heads_per_type = num_heads // self.num_specializations
        
        # 专业化特定的初始化
        self.specialized_inits = {
            'local': {'temperature': 0.5, 'query_bias': 'near'},
            'global': {'temperature': 2.0, 'query_bias': 'far'},
            'boundary': {'temperature': 1.0, 'query_bias': 'gradient'}
        }
```

#### 3.3 渐进式注意力训练
```python
class ProgressiveAttentionTraining:
    def __init__(self, total_epochs=200):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # 训练阶段定义
        self.phases = [
            {'epochs': 50, 'diversity_weight': 0.2, 'temperature': 'high'},
            {'epochs': 100, 'diversity_weight': 0.1, 'temperature': 'medium'},
            {'epochs': 50, 'diversity_weight': 0.05, 'temperature': 'low'}
        ]
    
    def get_current_config(self):
        # 根据当前epoch返回配置
        epoch_sum = 0
        for phase in self.phases:
            epoch_sum += phase['epochs']
            if self.current_epoch <= epoch_sum:
                return phase
        return self.phases[-1]
```

## 🧪 实验验证计划

### 实验组设计
| 实验组 | 改进措施 | 预期效果 | 优先级 |
|--------|----------|----------|--------|
| A1 | 温度参数优化 + 查询多样化 | 快速验证，立即改善 | 高 |
| A2 | A1 + 注意力多样性正则化 | 进一步改善多样性 | 高 |
| B1 | 多头独立温度 + 尺度特异性 | 增强尺度差异 | 中 |
| B2 | 可学习查询向量 | 更灵活的注意力模式 | 中 |
| C1 | 渐进式训练策略 | 稳定训练过程 | 低 |
| C2 | 完整架构改进 | 综合最优效果 | 低 |

### 评估指标
1. **注意力多样性得分**: 计算多头注意力的差异度
2. **尺度特异性**: 不同尺度注意力权重的空间差异
3. **训练稳定性**: 损失曲线和注意力熵的变化
4. **下游任务性能**: 基因表达重建误差
5. **可视化质量**: 注意力模式的空间可解释性

### 数据集
1. **DLPFC**: 标准Visium数据，7个空间域
2. **HBC**: VisiumHD高分辨率数据
3. **合成数据**: 用于控制实验验证

## 📊 实施路线图

### 第1周：快速修复和验证
- [ ] 实现温度参数优化
- [ ] 实现查询向量多样化
- [ ] 调整dropout策略
- [ ] 运行实验A1，评估效果
- [ ] 根据结果调整参数

### 第2周：正则化和中级改进
- [ ] 实现注意力多样性正则化
- [ ] 实现多头独立温度
- [ ] 运行实验A2和B1
- [ ] 分析尺度特异性改善
- [ ] 优化正则化权重

### 第3周：高级架构改进
- [ ] 实现可学习查询向量
- [ ] 实现注意力头专业化
- [ ] 运行实验B2和C1
- [ ] 进行消融实验
- [ ] 确定最佳配置

### 第4周：综合测试和优化
- [ ] 实现完整架构改进
- [ ] 运行实验C2
- [ ] 进行跨数据集验证
- [ ] 性能调优和参数搜索
- [ ] 编写最终报告

## 🔧 代码修改指南

### 核心文件修改
1. **`CPS/model.py`**:
   - `TeacherNicheAttention`类的`__init__`和`forward`方法
   - 温度参数初始化逻辑
   - 查询向量生成逻辑

2. **`CPS/cps.py`**:
   - `CPSTrainer`类的`fit`方法
   - 添加多样性正则化损失
   - 修改训练循环监控

3. **`CPS/config.py`**:
   - 添加新的配置参数
   - 温度参数范围设置
   - 正则化权重参数

### 新文件创建
1. **`CPS/attention_utils.py`**:
   - 注意力评估工具
   - 多样性计算函数
   - 可视化工具

2. **`CPS/attention_losses.py`**:
   - 多样性损失函数
   - 特异性损失函数
   - 组合损失函数

## 📈 预期效果

### 短期目标（1-2周）
1. 解决注意力退化问题
2. 实现明显的多头差异化
3. 提高注意力模式的可解释性
4. 保持或提高下游任务性能

### 长期目标（1个月）
1. 建立稳定的注意力训练框架
2. 提供可配置的注意力机制
3. 在多个数据集上验证效果
4. 发表方法改进论文

## ⚠️ 风险与缓解

### 风险1：改进可能影响模型性能
- **缓解**: 逐步实施，每个阶段都验证下游任务性能
- **监控**: 同时跟踪注意力质量和重建误差

### 风险2：增加模型复杂度
- **缓解**: 提供配置选项，允许用户选择复杂度
- **优化**: 确保新增参数有明确的理论依据

### 风险3：训练不稳定
- **缓解**: 实现渐进式训练策略
- **监控**: 添加训练过程可视化工具

## 🤝 协作建议

1. **版本控制**: 为每个实验组创建独立分支
2. **实验记录**: 使用MLflow或W&B记录实验
3. **代码审查**: 每个改进提交前进行代码审查
4. **定期同步**: 每周同步进展和问题

---

**最后更新**: 2025-12-27  
**版本**: 1.0  
**状态**: 实施计划  
**负责人**: 模型架构团队