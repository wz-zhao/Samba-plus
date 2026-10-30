# 🌟 Samba+: General and Accurate Salient Object Detection via a More Unified Mamba-based Framework   


<p align="center">
<a href="https://arxiv.org/abs/2602.01593">
  <img src="https://img.shields.io/badge/arXiv-2602.01593-b31b1b.svg?logo=arxiv">
</a>


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
- Samba+ weights：[[baidu](https://pan.baidu.com/s/1w7n2FuEo0R1hD-JE1JT-3w?pwd=3xxz)(3xxz)]; 
[![Google Drive](https://img.shields.io/badge/Google-Drive-green?logo=google-drive)](https://drive.google.com/file/d/1S-8RV9vJT5VLuFcRQ8OS9GLEvf31ziCo/view?usp=sharing)

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



## MACL progressive training

`train_macl_ddp.py` supports single-GPU and DDP continual learning:

| Stage | Maximum epochs | Current samples | Replay samples |
|---|---:|---|---|
| 1 | 80 | RGB (DUTS-TR) | None |
| 2 | 180 | One randomly selected auxiliary task: RGB-D, RGB-T, or RGB+flow | 10% RGB |
| 3 | 240 | RGB+depth+flow or RGB+depth+thermal | 10% RGB, 30% dual-modal |

Replay rates count **samples**, including when the batch size is not divisible
by ten. Integer rounding differs by at most a sample. Different modality schemas
are forwarded separately and their sample-weighted losses share the same optimizer
update. The shared encoder first loads RGB-pretrained VMamba weights; the RGB-only
stage bypasses cross-modal fusion. Stage lengths and replay rates are configurable.
These are configurable upper limits, not required training durations or epoch
counts specified by the method. By default, **five consecutive epochs without
validation improvement** end the current stage (`--stage_patience 5`). An equal
score counts as no improvement; `--stage_min_delta` defaults to zero. Stage 3
stops training under the same rule. Improvement resets the counter. The best
epoch's model, optimizer, scaler, replay buffers and random states are restored
before the next stage, which starts a fresh cosine learning-rate schedule.
The upper limit still applies if scores keep improving. For example,
`--epoch 90 --stage1_epochs 20 --stage2_epochs 40` sets caps of 20/40/30.

Patience is updated once per epoch using the mean S-measure over selected
validation datasets for learned modalities: RGB in Stage 1, RGB and dual-modal
tasks in Stage 2, and all selected tasks in Stage 3. Earlier tasks participate to
monitor forgetting; unlearned modalities do not affect switching. Use held-out
validation data for selection, and reserve final test data for reporting.
`--test_tasks` is the existing name of the evaluation selector; it does not create
a separate validation split. By default validation runs only at epoch end;
optional intermediate validation does not consume extra patience epochs.
`--stage_patience 0` disables early switching and permits checks without validation.
The default training batch size is 2 per process (also 2 per GPU under DDP).

For the configured `192.168.18.182` server, upload this source directory and run
`bash train_4gpu.sh` from it. The launcher uses four GPUs, full datasets, the
existing Sigma data and pretrained weights, and patience 5. Override `DATA_ROOT`,
`PYTHON_BIN` or `CUDA_VISIBLE_DEVICES` if needed. Extra arguments are forwarded;
for example, `bash train_4gpu.sh --epoch 90 --stage1_epochs 20 --stage2_epochs 40`
keeps early switching and sets stage caps to 20/40/30. Output goes to
`checkpoints/MACL_Training` beside the uploaded launcher.

The rehearsal buffers store dataset names, sample indices and individual loss
scores. Sampling is weighted toward high-loss examples; priorities refresh when
examples are replayed. Per-source capacity reservations prevent a dual-modal source
from being evicted entirely. Images are read and augmented on demand. Non-RGB RQ
receives the ground-truth saliency mask, uses finer quantization in the foreground,
and bounds each foreground perturbation to 5% of its channel's dynamic range by
default (`--rq_salient_max_delta`). RGB is kept as the anchor input. Validation
does not apply RQ or use labels as model inputs.
VDT visible-light images are converted from OpenCV BGR into RGB before augmentation
and ImageNet normalization; depth and thermal preprocessing are unchanged.

Use Python 3.10+ and install a matching PyTorch/torchvision build for your CUDA
runtime first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
# Compile the fast scan backend on a CUDA development machine with nvcc:
cd models/encoders/selective_scan
python -m pip install . --no-build-isolation
cd ../../..
```

The PyTorch scan fallback (`--scan_backend torch`) supports forward/backward
checks without the compiled extension. It is substantially slower and more memory
intensive than the CUDA kernel. Use `--scan_backend cuda` for full-size training.

Arrange data under one root (image paths in each list are relative to that dataset):

```text
DATA_ROOT/
  rgb_dataset/DUTS-TR/train.txt
  rgb_dataset/DUTS-TE/test.txt
  rgbd_dataset/train_DUT/train.txt
  rgbd_dataset/NJU2K/test.txt
  rgbd_dataset/NLPR/test.txt
  rgbt_dataset/VT_train/train.txt
  rgbt_dataset/VT5000/test.txt
  vsod_dataset/train/{DAVIS,DAVSOD,FBMS}/train.txt
  vsod_dataset/test/DAVIS/test.txt
  dvsod_dataset/{RDVS,DVisal,vidsod_100}/train.txt
  dvsod_dataset/RDVS/test.txt
  VDT-2048/{Train,Test}/{V,D,T,GT}/
```

Defaults include additional test datasets: DUT-OMRON, VT821 and SegTrack-V2.
`--train_tasks` and `--test_tasks` select the datasets to load. Unselected datasets
are never initialized. Set `--vdt_train_root`/`--vdt_test_root` if VDT is elsewhere.
The legacy DVisal cross-dataset path mapping is preserved; the lists must match
the actual downloaded data. Data and checkpoint files are not bundled.

```bash
# Check only the selected dataset paths, without allocating the network:
python train_macl_ddp.py --data_root /path/to/data --check_data

# Single GPU, all three MACL stages:
python train_macl_ddp.py --data_root /path/to/data \
  --backbone_path /path/to/vssmsmall_dp03_ckpt_epoch_238.pth --scan_backend cuda

# Multiple GPUs; global rank is used for sampling, local rank for devices:
torchrun --standalone --nproc_per_node=4 train_macl_ddp.py \
  --data_root /path/to/data --backbone_path /path/to/vssmsmall_dp03_ckpt_epoch_238.pth \
  --scan_backend cuda

# Select a smaller set while retaining all three stage types:
python train_macl_ddp.py --data_root /path/to/data \
  --backbone_path /path/to/vssmsmall_dp03_ckpt_epoch_238.pth \
  --train_tasks DUTS-TR train_DUT DVSOD_train --test_tasks DUTS-TE NJU2K RDVS

# Short check on real data: one epoch per stage, at most 10 steps each:
python train_macl_ddp.py --data_root /path/to/data \
  --backbone_path /path/to/vssmsmall_dp03_ckpt_epoch_238.pth \
  --epoch 3 --stage1_epochs 1 --stage2_epochs 1 --batch_size 2 \
  --train_tasks DUTS-TR train_DUT VDT --test_tasks DUTS-TE NJU2K VDT \
  --max_train_samples 30 --max_test_samples 3 --max_steps_per_epoch 10 \
  --vdt_train_root /path/to/VDT/Train --vdt_test_root /path/to/VDT/Test

# Resume with the same data, stage lengths, epoch count, replay settings and world size:
python train_macl_ddp.py --data_root /path/to/data \
  --resume /path/to/checkpoints/MACL_Training/last_checkpoint.pth
```

`--from_scratch` explicitly disables RGB pretraining for experiments; this differs
from the pretrained Stage-1 setup above. `--model_path` optionally loads a SOD
warm start. Missing or incompatible explicit checkpoints fail rather than silently
falling back to random initialization. Use `--mode test --model_path ...` for
standalone evaluation, with `--test_tasks` selecting its datasets.

The runner evaluates at each epoch end and optionally every N steps. Distributed
evaluation uses disjoint shards without padded duplicates. `last_checkpoint.pth`
is updated every epoch, including patience counters and the active stage.
`stage1_best.pth`, `stage2_best.pth` and `stage3_best.pth` are full best-epoch
snapshots; corresponding `stageN_best_model.pth` files contain weights only.
The files are written under `SAVE_FOLD/MACL_Training/`; `stage_summary.json`
records actual stage durations, best epochs and stopping reasons.
`overall_best.pth` contains the latest completed stage's best weights after a
transition (during training it may also track intermediate validation bests).
Standalone epoch weights are saved at boundaries and every `--epoch_save` epochs.
When moving a resume bundle, keep the current stage's `stageN_best.pth` beside
`last_checkpoint.pth`; it is needed to restore the best epoch on a later transition.
Resume also requires unchanged validation selection, patience and stage caps.
Legacy checkpoints without stage progress must use `--model_path` for a weights-only
warm start, which restarts the three-stage schedule and creates new replay buffers.
Full checkpoints also store SHA-256 fingerprints of the ordered training sample
mappings, manifest file contents, dataset lengths and subset selections. On
`--resume`, changed mappings or `--max_train_samples` are rejected before model
allocation, and replay indices are checked against the effective dataset lengths.
DDP workers must agree on these identities. This validates metadata and index
identity without hashing the image file payloads. Keep original lists, paths and
subset settings when resuming. Older checkpoints without these fingerprints
cannot safely restore replay buffers; use `--model_path` for a weights-only warm
start that creates new buffers instead of bypassing the check.

In this Windows workspace, the verified environment is `conda activate samba`.
The default base Python is a different environment. CPU checks use `--device cpu`.
The installed Windows PyTorch nightly has no usable Gloo transport; use Linux/WSL
for multi-process DDP. The two-worker DDP smoke test was verified in an isolated
WSL CPU environment. CUDA AMP forward/backward was checked in the Windows
`samba` environment using the PyTorch scan backend.

On 2026-10-30, short runs on the user's Linux server also passed using the existing
CUDA scan extension, RGB-pretrained VMamba, real DUTS-TR/train_DUT/VDT data and
448 x 448 inputs. Both single-GPU and two-GPU NCCL DDP completed all three stages,
10 steps per stage, including validation and checkpoints. These are execution
checks on small subsets, not full-training accuracy or convergence results.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m tests.ddp_smoke
```
