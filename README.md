# 🌟 Samba+: General and Accurate Salient Object Detection via a More Unified Mamba-based Framework
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_show_1.png" width="90%">


<p align="center">
  Wenzhuo Zhao, Keren Fu, Jiahao He, Xiaohong Liu, Qijun Zhao, Guangtao Zhai
</p>

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



---


## 🚀 Introduction

-  **Samba** and **Samba+** are the first adaptation of **State Space Models (SSMs)** to **Salient Object Detection (SOD)** tasks and demonstrate their strong capability in modeling long-range dependencies across multiple modalities.

-  **Samba+** is also the **first truly versatile SOD model** in the community.

---


## 🔑 Motivation 

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/Fig_intro_1.png" width="70%">
</p>

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_sns_1.png" width="70%">
</p>

- 🧠 The first **Mamba-based architecture** for SOD  
- 🎯 By rethinking Mamba’s **scanning strategy** in the context of SOD, we introduce a **saliency-guided Mamba block (SGMB)** equipped with a **spatial neighborhood scanning (SNS)** algorithm, enabling better modeling of spatially coherent salient structures.
- 🌈 Support for **RGB SOD / RGB-D SOD / RGB-T SOD / VDT SOD / VSOD / RGB-D VSOD** via a **single versatile** model

---

## 🧩 Framework Overview

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overview_1.png" width="85%">
</p>
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overview_new_1.png" width="85%">
</p>

---

## 📂 Datasets

| Task | Training Datasets | Testing Datasets | Download |
|------|-------------------|------------------|----------|
| RGB SOD | DUTS | DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-O | [Baidu](https://pan.baidu.com/s/1oljb1_kkUH7rhWZCy8ic4g?pwd=x7kn) (`x7kn`) |
| RGB-D SOD | NJU2K, NLPR, DUT-RGBD | NJU2K, NLPR, DUT-RGBD, SIP, STERE | [Baidu](https://pan.baidu.com/s/1ibrO3CS7rn7bJUAy8hM9mQ?pwd=8b9c) (`8b9c`) |
| RGB-T SOD | VT5000 | VT5000, VT821, VT1000 | [Baidu](https://pan.baidu.com/s/1PKW5d_Yr5NFEnq9Q82HitA?pwd=xhrm) (`xhrm`) |
| VDT SOD | VDT-2048 | VDT-2048 | [Baidu](https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy) (`bipy`) |
| VSOD | DAVIS, DAVSOD, FBMS | DAVIS, DAVSOD, FBMS, Seg-V2, VOS | [Baidu](https://pan.baidu.com/s/1zQ-vuDnSfRzJ1T_T-hh7sA?pwd=kcmu) (`kcmu`) |
| RGB-D VSOD | RDVS, DVisal, Vidsod_100 | RDVS, DVisal, Vidsod_100 | [Baidu](https://pan.baidu.com/s/1VRL3jk7AsQCkL26hwg1rZA?pwd=q9ty) (`q9ty`) |

### 🛠️ Overlapping Samples
To avoid data leakage and ensure fair training, we only retain the samples from DVisal together with their ground-truth annotations.
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_overlap_1.png" width="70%">
</p>

---

## ✨  Visual Results
All evaluated saliency maps are put here: [Baidu](https://pan.baidu.com/s/1Lv9P8JW3YyI6Ds76wUnXaQ?pwd=p2h2)(`p2h2`)
<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/fig_visual_1.png" width="100%">
</p>

### Other Tasks that Emphasize Spatial Continuity

<p align="center">
  <img src="https://github.com/wz-zhao/Samba-plus/blob/main/Figures/other.jpg" width="100%">
</p>

---

## ⚙️ Environment Setup

- PyTorch 1.13.1
- CUDA 11.7
- VMamba-S backbone weights：[[baidu](https://pan.baidu.com/share/init?surl=SaEV237VCzSEn558gEBiXg)(zsxa)]
- Samba+ weights：[[baidu](https://pan.baidu.com/s/1w7n2FuEo0R1hD-JE1JT-3w?pwd=3xxz)(3xxz)]



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

