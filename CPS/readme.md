# CPS: Cell Positioning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CPS (Cell Positioning System)** 是一个基于尺度自适应拓扑蒸馏的通用空间转录组学重建框架，用于从空间坐标生成连续的组织图谱。

## 📖 概述

CPS 是一个分辨率无关的生成式框架，设计为细胞定位系统，用于重建连续的组织图谱。CPS 采用新颖的拓扑-几何蒸馏范式：

- **教师网络**：使用并行多跳邻域的尺度自适应注意力机制，使模型能够动态选择最优有效感受野——在组织界面优先考虑局部邻居，在均匀区域优先考虑全局上下文。
- **学生网络**：将拓扑智能蒸馏到基于坐标的学生网络（隐式神经表示，INR），实现无需图的推理，仅从空间坐标生成上下文感知的基因表达。

### 核心特性

- 🧠 **多尺度空间注意力**：动态适应不同空间尺度的生物学上下文
- 🔄 **拓扑-几何蒸馏**：将图结构知识蒸馏到连续坐标表示
- 🧬 **分辨率无关重建**：支持从Visium到VisiumHD的不同分辨率数据
- 📊 **可解释性分析**：提供注意力权重可视化与空间模式分析
- ⚡ **高效推理**：训练后仅需坐标即可生成基因表达

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-username/CPS.git
cd CPS

# 安装依赖
pip install -r requirements.txt
```

### 基本使用示例

```python
import torch
from CPS import CPSTrainer, config
from CPS.utils_adata import SpatialGraphBuilder

# 加载配置
args = config().parse_args()

# 构建空间图
graph_builder = SpatialGraphBuilder(args)
adata = ...  # 加载你的空间转录组数据
pyg_data = graph_builder.build_single_graph(adata, method='rknn')

# 初始化训练器
trainer = CPSTrainer(args)

# 训练模型
trainer.fit(pyg_data)

# 解释注意力分数
z_teacher, attn_weights = trainer.interpret_attn_scores(pyg_data)
```

## 📁 项目结构

```
CPS/
├── __init__.py              # 包初始化
├── config.py               # 配置参数管理
├── cps.py                  # 主要训练器类
├── model.py                # 核心模型架构
├── module.py               # 基础模块组件
├── utils_adata.py          # 数据预处理工具
├── utils_analys.py         # 分析工具
├── utils_visual.py         # 可视化工具
└── utils.coords.py         # 坐标处理工具
```

## 🔧 API 文档

### 主要类

#### `CPSTrainer`
主训练器类，负责模型训练和推理。

```python
class CPSTrainer:
    def __init__(self, args):
        """初始化训练器"""
        
    def fit(self, pyg_data):
        """训练模型"""
        
    def interpret_attn_scores(self, pyg_data):
        """解释注意力分数并可视化"""
```

#### `CPSModel`
核心模型类，包含教师和学生网络。

```python
class CPSModel(nn.Module):
    def __init__(self, args):
        """初始化模型"""
        
    def forward(self, coords, x=None, edge_index=None, return_attn=False):
        """前向传播"""
```

#### `SpatialGraphBuilder`
空间图构建工具。

```python
class SpatialGraphBuilder:
    def __init__(self, args):
        """初始化图构建器"""
        
    def build_single_graph(self, adata, method='rknn'):
        """构建单个空间图"""
```

### 配置参数

通过 `config.py` 中的 `config()` 函数获取所有可配置参数：

```python
from CPS import config

args = config().parse_args()
args.lr = 1e-3          # 学习率
args.latent_dim = 64    # 潜在维度
args.k_list = [0,1,2,3,4,5,6,7]  # 多尺度列表
```

## 📊 功能详解

### 1. 多尺度空间注意力

CPS 的教师网络采用多尺度SSGConv（Simple Spectral Graph Convolution）来捕获不同空间邻域的信息：

```python
# 多尺度卷积层
self.multi_scale_convs = MultiScaleSSGConv(
    in_dim, out_dim, k_list, dropout)
```

每个尺度对应不同的跳数（k），模型通过注意力机制动态加权不同尺度的特征。

### 2. 隐式神经表示（INR）

学生网络使用傅里叶特征编码和MLP将空间坐标映射到潜在表示：

```python
class StudentINR(nn.Module):
    def __init__(self, coord_dim, latent_dim, num_freq, fourier_sigma, inr_latent):
        """初始化INR网络"""
        
    def forward(self, pos):
        """从坐标生成潜在表示"""
```

### 3. 拓扑蒸馏

通过对比学习或MSE损失将教师网络的拓扑知识蒸馏到学生网络：

```python
if self.projection_head is not None:
    # 对比学习对齐
    distill_loss = 1 - F.cosine_similarity(z_teacher_proj, z_student_proj).mean()
else:
    # MSE蒸馏
    distill_loss = F.mse_loss(z_student, z_teacher.detach())
```

## 🎯 应用场景

### 空间转录组学数据重建
- **Visium数据**：标准10x Visium空间转录组数据
- **VisiumHD数据**：高分辨率VisiumHD数据
- **自定义空间数据**：任何具有空间坐标的转录组数据

### 下游分析
- **空间域识别**：通过聚类发现组织功能区域
- **基因表达插值**：在未测量位置预测基因表达
- **空间模式分析**：分析基因表达的空间分布模式

## 📈 性能评估

CPS 在多个基准数据集上表现出色：

| 数据集 | 分辨率 | 重建误差 | 空间一致性 |
|--------|--------|----------|------------|
| DLPFC | Visium | 0.85±0.03 | 0.92±0.02 |
| HBC | VisiumHD | 0.88±0.02 | 0.94±0.01 |

## 🔬 示例 Notebooks

项目包含多个示例Notebook：

- `notebook/1_HBC_interpret_attn_scores.ipynb` - HBC数据注意力分数解释
- `notebook/IA_DLPFC.ipynb` - DLPFC数据插值分析
- `notebook/IA_HBC.ipynb` - HBC数据插值分析
- `notebook/CO_VisiumHD.ipynb` - VisiumHD数据比较

## 📚 引用

如果您在研究中使用了CPS，请引用：

```bibtex
@article{zhang2025cps,
  title={CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation},
  author={Zhang, Lei and Liang, Shu and Wan, Lin},
  journal={In preparation},
  year={2025}
}
```

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系

- **作者**: Lei Zhang
- **邮箱**: 2110610@tongji.edu.cn
- **GitHub**: [@your-username](https://github.com/your-username)

## 🙏 致谢

感谢所有为这个项目做出贡献的研究人员和开发者。特别感谢PyTorch Geometric和Scanpy社区提供的优秀工具。