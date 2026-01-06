- 12.12
> init code framework and schedule

---

- 12.13-15
> model framework: 1. multi-scale neighborhood cross attention 2. INR model 3. decoder 4. distilled module

- **! problem**
1. preprocess for large graph: first to propagate, second to random walk (1114 task)
2. INR activation function
3. distilled module

---

- 12.16
> SSGConv study and research.

---

- 12.17
> code framework and ref code of stage

- **! ToDo**
1. stage code and analysis - todo
2. cps code framework - finish

---

- 12.18
> DLPFC experiment complete: SI, GI, DeNoise
> Large scale data preprocess (multi-scale graph data)

---

12.22
---
> Benchmark SI/GI/DN tasks
1. STAGE/SUICA prepare data
2. benchmark pipeline
3. compute metrics and optimize the parameters

> interpret the attention scores
1. R/H scores //ok
2. trajectory 
3. visualization



12.23
---
- banchmark of CPS
    - spatial imputation
    - gene imputation

- interpretability of attention scores
    - boundary is not clear
    - domain identification is not good
- we need'nt to compute the domain identification, attention scores are enough to prove the interpretability.

12.24
---
1. interpertability of attention scores
    - single cell/subcellular resoulved dataset attention scores
    - 
2. benchmark of spatial imputation and gene imputation
    - STAGE/SUICA code framework
    - benchmark on DLPFC

**key**: INR infers the transition using spot with composition of pure tumor or mixed cells

1.6 review code
1. review code
2. rethink the framework
3. determine the model