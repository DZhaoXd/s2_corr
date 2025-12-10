# ✨ S²-Corr: State-Space Correlation Refinement for Open-Vocabulary Domain Generalization

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-11.7-76b900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" />
  <img src="https://img.shields.io/badge/Model-EVA--CLIP-green?logo=openai" />
</p>

**Official implementation of**
 *S²-Corr: State-Space Correlation Refinement for Open-Vocabulary Domain Generalization in Semantic Segmentation*

S²-Corr introduces a state-space powered correlation refinement module that stabilizes text–image alignment under domain shift, achieving SOTA performance on both Real-to-Real and Synthetic-to-Real OVDG-SS settings.

---

## 🚀 Features

* 🧩 **State-Space Correlation Aggregation**
  Robust long-range correlation modeling via scan-based state passing.

* 🔍 **Open-Vocabulary Semantic Segmentation**
  Compatible with EVA-CLIP text/image encoders.

* 🌍 **Domain Generalization**
  Train on CS-7 / GTA-7 → test on ACDC / BDD / Mapillary / ROADWork.

* 🎯 **Supports Multiple Category Spaces** (7 / 19 / 30 / 41 / 58 classes)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DZhaoXd/s2_corr.git
cd s2_corr
```

### 2. Create Conda Environment

```bash
conda create -n S2_Corr python=3.8
conda activate S2_Corr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements.txt
pip install -e .
```


## 📁 Dataset Preparation

### 🔧 Conversion Scripts

####  7- & 19-Class Format

```bash
python tools/convert_datasets_to19/gta.py data/gta
python tools/convert_datasets_to19/cityscapes.py data/cityscapes
python tools/convert_datasets_to19/mapillary.py data/mapillary
python tools/convert_datasets_ovss/prepare_cityscapes_seen_7.py
python tools/convert_datasets_ovss/process_GTA_19_to_7.py
```

####  large-Vocabulary (7 / 30 / 41 / 58)

```bash
python cp_Mapi_training.py   # merge train set and val set 
python tools/convert_datasets_ovss/process_Mapi_65.py
python tools/convert_datasets_ovss/process_RW_10.py
```


Folder structure under `data/` should look like:
```
data/
├── GTA5/
│   └── GTAV/
│       ├── images/                 # 24,999
│       ├── labels_7/               # c-7
│       └── labels_19/              # c-19
│
├── cityscape/
│   ├── leftImg8bit/
│   │   └── train/                  # 2,899
│   ├── gtFine_7/
│   │   └── train/                  # c-7
│   └── gtFine_19/
│       └── train/                  # c-19
│
├── BDD/
│   ├── bdd100k/
│   │   ├── images/10k/val/         # 1,000
│   │   └── labels/sem_seg/masks/val/   # c-19
│   │
│   ├── bdd_inpaint41/
│   │   ├── images/                 # 1,000
│   │   └── labels/                 # c-41
│   
│
├── ACDC/
│   ├── rgb_anon/train/             # 1,600 (c-19)
│   └── gt/train/                   # c-19
│   │
│   └── ACDC_inpaint41/
│       ├── images/                 # 1,000
│       └── labels/                 # c-41
│
│
├── mapillary/
│   ├── val/
│   │   ├── images/                 # 2,000
│   │   └── labels_TrainIds/        # c-19
│   │
│   └── OV_30/
│       ├── images/                 # 3,943
│       └── labels/                 # c-30
│
└── ROADWork_Data/
    ├── images/                     # 2,098
    └── gtFine_10/                  # c-10

```
---

## 🧠 Pretrained EVA-CLIP Models

Download EVA-CLIP weights from:

👉 [https://github.com/baaivision/EVA/tree/master/EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)

Place under:

```
Pretrain/
  EVA02_CLIP_B_psz16_s8B.pt
  EVA02_CLIP_L_336_psz14_s6B.pt
```

---

## Training

Training script format:

```bash
bash run.sh <CONFIG_YAML> <NUM_GPUS> <OUTPUT_DIR>
```

### 🔹 CS-7 (Real-to-Real)

**ViT-B/16**

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512 \
  > logs/cs7_eva_b16_r512.log 2>&1 &
```

**ViT-L/14**

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh configs/cs7_catseg_vitl.yaml 1 outputs/cs7_eva_L14_r448 \
  > logs/cs7_eva_L14_r448.log 2>&1 &
```

---

### 🔹 GTA-7 (Synthetic-to-Real)

**ViT-B/16**

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh configs/gta5_seen7_catseg.yaml 1 outputs/gta_seen7_eva_b16_r512 \
  > logs/gta_seen7_eva_b16_r512.log 2>&1 &
```

**ViT-L/14**

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh configs/gta5_seen7_catseg_vitl.yaml 1 outputs/gta_seen7_eva_L14_r448 \
  > logs/gta_seen7_eva_L14_r448.log 2>&1 &
```

---

### 🔹 CS-19 (19 Classes)

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh configs/cs19_catseg_vitl.yaml 1 outputs/cs19_eva_L14_r448 \
  > logs/cs19_eva_L14_r448.log 2>&1 &
```

---

## 👀 Visualization

### 🎨 Segmentation Masks

```bash
CUDA_VISIBLE_DEVICES=0 nohup sh demo/vis.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512 \
  > logs/viz_cs7_eva_b16_r512.log 2>&1 &
```

###  Correlation / Attention Maps

```bash
CUDA_VISIBLE_DEVICES=0 nohup sh demo/vis_atten.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512 \
  > logs/viz_attention_cs7_eva_b16_r512.log 2>&1 &
```

---

## 📚 Citation

```bibtex
@article{zhao2026s2corr,
  title={S²-Corr: State-Space Correlation Refinement for Open-Vocabulary Domain Generalization},
  year={2026}
}
```

---

## ❤️ Acknowledgements

This project builds upon CAT-Seg

