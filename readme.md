CPS: A Cell Positioning System for Universal Spatial Transcriptomics Reconstruction via Scale-Adaptive Topological Distillation
===
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


📖 Overview
---

![](https://github.com/tju-zl/CPS/blob/main/overview_cps.png)
We introduce CPS, a resolution-independent generative framework designed as a Cell Positioning System to reconstruct continuous tissue maps. CPS employs a novel topological-geometric distillation paradigm. The teacher network utilizes a scale-adaptive attention mechanism over parallel multi-hop neighborhoods, enabling the model to dynamically select the optimal effective receptive field—prioritizing local neighbors at tissue interfaces and global contexts in homogeneous regions. This topological intelligence is distilled into a coordinate-based Student network (Implicit Neural Representation), allowing for graph-free inference that generates context-aware gene expression solely from spatial coordinates.

🚀 Reproduction of results
---

We provide comprehensive Jupyter notebooks to reproduce the results and figures presented in the paper. The experiments are organized into three main directories: `benchmark`, `case_study`, and `Interpret`.

* **📂 benchmark/**: Contains benchmarks for Spatial Imputation and Gene Imputation on the 12-slice DLPFC dataset.
    * `1_spatiual_imputation_DLPFC.ipynb`: Benchmarking spatial imputation performance on #151673 .
    * `2_gene_imputation_DLPFC.ipynb`: Benchmarking gene imputation performance on #151673.
    * `3_DLPFC_12_SI.ipynb` & `4_DLPFC_12_GI.ipynb`: Detailed benchmarking scripts for the 12 DLPFC slices.

* **📂 case_study/**: Demonstrates Super-Resolution (SR) capabilities and Scalability analysis.
    * `1_MBSP_SR.ipynb`: **Super-Resolution Task**. Reconstructs high-fidelity gene expression at arbitray resolution using MBSP data.
    * `2_HD_Scalable.ipynb`: **Scalability on Visium HD**. Demonstrates efficient processing of Visium HD data.
    * `3_MED_Efficient_*.ipynb`: **Efficient on Mouse Embryo**. Validates performance and efficiency across varying data scales using the Mouse Developing Embryo (MED) atlas.
    * `time_compute.ipynb`: Analysis of computational time and resource usage.

* **📂 Interpret/**: Focuses on model interpretability.
    * `1_HBC_interpret_attn_scores.ipynb`: **Interpretability Analysis**. Visualizes multi-scale attention scores on the Human Breast Cancer (HBC) dataset to decode tissue heterogeneity.

- To run these notebooks, ensure you have installed the required dependencies and downloaded the necessary datasets.

📈 SRT data for evaluating CPS
---
The datasets analyzed in this study are publicly available from their respective repositories:

- The human Dorsolateral Prefrontal Cortex (DLPFC) dataset is available via the spatialLIBD package or at http://research.libd.org/spatialLIBD/.
- The Human Breast Cancer, Mouse Posterior Brain (Visium) and Mouse Brain Visium HD datasets can be downloaded from the 10x Genomics website https://www.10xgenomics.com/resources/datasets.
- The Stereo-seq Mouse Embryo Atlas is accessible through the MOSTA database https://db.cngb.org/stomics/mosta/.

🤝 Software depdendencies
---
- scanpy==1.9.8
- scikit-learn==1.3.2
- scipy==1.10.1
- squidpy==1.2.3
- torch==2.1.0+cu121
- torch_geometric==2.5.3
- transformers==4.46.3


---