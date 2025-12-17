# 🌟 Samba+: General and Accurate Salient Object Detection via a More Unified Mamba-based Framework


<p align="center">
  Wenzhuo Zhao, Keren Fu, Jiahao He, Xiaohong Liu, Qijun Zhao, Guangtao Zhai
</p>

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_intro.pdf" width="100%">
</p>

---

## 📌 Conference Version (CVPR 2025 Highlight)

<p align="center">
  <strong>Samba: A Unified Mamba-based Framework for General Salient Object Detection</strong>
</p>

<p align="center">
  <a href="https://www.kerenfu.top/sources/CVPR2025_Samba.pdf">
    <img src="https://img.shields.io/badge/Paper-CVPR%202025 Highlight-blue">
  </a>
  <a href="https://github.com/Jia-hao999/Samba/blob/main/CVPR2025_Samba_Chinese.pdf">
    <img src="https://img.shields.io/badge/中文版-PDF-red">
  </a>
  <a href="https://github.com/Jia-hao999/Samba">
    <img src="https://img.shields.io/badge/Model-Samba-black?logo=github">
  </a>
</p>


</p>


---


## 🚀 Introduction

**Samba** is the first **unified Mamba-based framework** for **General Salient Object Detection (SOD)**.  
We pioneer the adaptation of **State Space Models (SSMs)** to SOD tasks and demonstrate their strong capability in modeling long-range dependencies across multiple modalities.

### 🔑 Key Contributions

- 🧠 Pure **Mamba-based architecture** for unified SOD modeling  
- 🎯 **Saliency-Guided Mamba Block (SGMB)** with Spatial Neighboring Scanning (SNS)  
- 🔄 **Context-Aware Upsampling (CAU)** for hierarchical feature alignment  
- 🌈 Support for **RGB SOD / RGB-D SOD / RGB-T SOD / VDT SOD / VSOD / RGB-D VSOD**

---

## 🧩 Framework Overview

<p align="center">
  <img src="https://github.com/Jia-hao999/Samba/blob/main/Figure/overview.png" width="85%">
</p>

---

## 📂 Datasets

| Task | Training Datasets | Testing Datasets | Download |
|------|-------------------|------------------|----------|
| RGB SOD | DUTS | DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-O | [Baidu](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g) (`x7kn`) |
| RGB-D SOD | NJU2K, NLPR, DUT-RGBD | NJU2K, NLPR, DUT-RGBD, SIP, STERE | [Baidu](https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ) (`8b9c`) |
| RGB-T SOD | VT5000 | VT5000, VT821, VT1000 | [Baidu](https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA) (`xhrm`) |
| VDT SOD | VDT-2048 | VDT-2048 | [Baidu](https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy) (`bipy`) |
| VSOD | DAVIS, DAVSOD, FBMS | DAVIS, DAVSOD, FBMS, Seg-V2, VOS | [Baidu](https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA) (`kcmu`) |
| RGB-D VSOD | RDVS, DVisal, Vidsod_100 | RDVS, DVisal, Vidsod_100 | [Baidu](https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA) (`q9ty`) |

---
## ⚙️ Environment Setup

- PyTorch 1.13.1
- CUDA 11.7



## 📚 Citation

If you find this repository useful, please use the following BibTeX entry for citation and give us a star⭐.

```bibtex

@InProceedings{He_2025_CVPR,
  author    = {He, Jiahao and Fu, Keren and Liu, Xiaohong and Zhao, Qijun},
  title     = {Samba: A Unified Mamba-based Framework for General Salient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {25314--25324}
}

