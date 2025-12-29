CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation
===
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


📖 Overview
---
We introduce CPS, a resolution-independent generative framework designed as a Cell Positioning System to reconstruct continuous tissue maps. CPS employs a novel topological-geometric distillation paradigm. The teacher network utilizes a scale-adaptive attention mechanism over parallel multi-hop neighborhoods, enabling the model to dynamically select the optimal effective receptive field—prioritizing local neighbors at tissue interfaces and global contexts in homogeneous regions. This topological intelligence is distilled into a coordinate-based Student network (Implicit Neural Representation), allowing for graph-free inference that generates context-aware gene expression solely from spatial coordinates.

🚀 Getting Started
---
See tutorials

📈 SRT data for evaluating CPS
---
All datasets are open access.

🤝 Software depdendencies
---
- scanpy==1.9.8
- scikit-learn==1.3.2
- scipy==1.10.1
- squidpy==1.2.3
- torch==2.1.0+cu121
- torch_geometric==2.5.3
- transformers==4.46.3

📚 Cite
---
```bibtex
@article{zhang2025cps,
  title={CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation},
  author={Zhang, Lei and Liang, Shu and Wan, Lin},
  journal={In preparation},
  year={2025}
}
```