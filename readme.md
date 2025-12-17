CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation
===

Overview
---
**Motivation**: Deciphering tissue heterogeneity from spatial transcriptomics (ST) data requires balancing two conflicting objectives: integrating broad spatial contexts for robust denoising and preserving local fidelity for sharp boundary definition. Existing graph-based methods often struggle with this trade-off, where fixed-scale aggregation leads to over-smoothing of fine-grained structures. Moreover, the reliance on discrete graph topology limits their ability to model continuous biological manifolds and hinders scalable inference for emerging sub-cellular resolution technologies.

**Method**: We introduce CPS, a resolution-independent generative framework designed as a Cell Positioning System to reconstruct continuous tissue maps. CPS employs a novel topological-geometric distillation paradigm. The teacher network utilizes a scale-adaptive attention mechanism over parallel multi-hop neighborhoods, enabling the model to dynamically select the optimal effective receptive field—prioritizing local neighbors at tissue interfaces and global contexts in homogeneous regions. This topological intelligence is distilled into a coordinate-based Student network (Implicit Neural Representation), allowing for graph-free inference that generates context-aware gene expression solely from spatial coordinates.

**Results**: Extensive evaluations on both 10x Visium and sub-cellular Stereo-seq datasets demonstrate that CPS achieves state-of-the-art performance in super-resolution, imputation, and denoising tasks. Beyond reconstruction, CPS offers unique interpretability: the attention scores function as a Niche Radar, automatically highlighting cell contours and tissue boundaries in high-resolution data without supervision. By bridging discrete graph topology and continuous geometry, CPS provides a robust and versatile tool for resolving spatial heterogeneity across varying scales.

Getting Started
---
See tutorials

SRT data for evaluating CPS
---
All datasets are open access.

Software depdendencies
---
- scanpy==1.9.8
- scikit-learn==1.3.2
- scipy==1.10.1
- squidpy==1.2.3
- torch==2.1.0+cu121
- torch_geometric==2.5.3
- transformers==4.46.3

In preparation
---
Lei Zhang, Shu Liang, Lin Wan, CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation.