# 🌟 Samba+  
## General and Accurate Salient Object Detection via a Unified Mamba-based Framework

<p align="center">
  <strong>Wenzhuo Zhao</strong>, Keren Fu, Jiahao He, Xiaohong Liu, Qijun Zhao, Guangtao Zhai
</p>

<p align="center">
  <a href="https://www.kerenfu.top/sources/CVPR2025_Samba.pdf">📄 Paper</a> |
  <a href="https://github.com/Jia-hao999/Samba/blob/main/CVPR2025_Samba_Chinese.pdf">📘 中文版</a> |
  <strong>CVPR 2025 (Highlight)</strong>
</p>

<p align="center">
  <img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/intro_1.png" width="80%">
</p>

---

## 🚀 Introduction

**Samba** is the first **unified Mamba-based framework** for **General Salient Object Detection (SOD)**.  
We pioneer the adaptation of **State Space Models (SSMs)** to SOD tasks and demonstrate their strong capability in modeling long-range dependencies across multiple modalities.

### 🔑 Key Contributions

- 🧠 Pure **Mamba-based architecture** for unified SOD modeling  
- 🎯 **Saliency-Guided Mamba Block (SGMB)** with Spatial Neighboring Scanning (SNS)  
- 🔄 **Context-Aware Upsampling (CAU)** for hierarchical feature alignment  
- 🌈 Support for **RGB / RGB-D / RGB-T / VSOD / RGB-D VSOD**

---

## 🧩 Framework Overview

<p align="center">
  <img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/overview.png" width="85%">
</p>

---

## ⚙️ Environment Setup

```bash
PyTorch 1.13.1
CUDA 11.7
