# 🌟 Samba+  
## General and Accurate Salient Object Detection via a Unified Mamba-based Framework

<p align="center">
  Wenzhuo Zhao, Keren Fu, Jiahao He, Xiaohong Liu, Qijun Zhao, Guangtao Zhai
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

## 📂 Data Preparation

### 1️⃣ RGB Salient Object Detection (RGB SOD)

**Training Dataset**
- DUTS (Train)

**Testing Datasets**
- DUTS  
- ECSSD  
- HKU-IS  
- PASCAL-S  
- DUT-O  

📎 Dataset Download:  
[https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g)  
Extraction Code: `x7kn`

---

### 2️⃣ RGB-D Salient Object Detection (RGB-D SOD)

**Training Datasets**
- NJU2K  
- NLPR  
- DUT-RGBD  

**Testing Datasets**
- NJU2K  
- NLPR  
- DUT-RGBD  
- SIP  
- STERE  

📎 Dataset Download:  
[https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ](https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ)  
Extraction Code: `8b9c`

---

### 3️⃣ RGB-T Salient Object Detection (RGB-T SOD)

**Training Dataset**
- VT5000  

**Testing Datasets**
- VT5000  
- VT821  
- VT1000  

📎 Dataset Download:  
[https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA](https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA)  
Extraction Code: `xhrm`

---

### 4️⃣ Video Salient Object Detection (VSOD)

**Training Datasets**
- DAVIS  
- DAVSOD  
- FBMS  

**Testing Datasets**
- DAVIS  
- DAVSOD  
- FBMS  
- Seg-V2  
- VOS  

📎 Dataset Download:  
[https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA](https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA)  
Extraction Code: `kcmu`

---

### 5️⃣ RGB-D Video Salient Object Detection (RGB-D VSOD)

**Training Datasets**
- RDVS  
- DVisal  
- Vidsod_100  

**Testing Datasets**
- RDVS  
- DVisal  
- Vidsod_100  

📎 Dataset Download:  
[https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA](https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA)  
Extraction Code: `q9ty`


## ⚙️ Environment Setup

```bash
PyTorch 1.13.1
CUDA 11.7


## 📚 Citation

If you use **Samba** in your research or find this work helpful, please consider citing:

```bibtex
@InProceedings{He_2025_CVPR,
  author    = {He, Jiahao and Fu, Keren and Liu, Xiaohong and Zhao, Qijun},
  title     = {Samba: A Unified Mamba-based Framework for General Salient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {25314--25324}
}

