# CIFAKE Retraining Guide

This guide walks you through downloading the CIFAKE dataset and retraining the
`ImageDeepfakeModel` (EfficientNet-B0) image classifier used by the Multimodal AV
Fraud & Deepfake Detection API.

---

## Why Retraining Is Needed

The existing checkpoint (`efficientnet_b0_image_best.pth`) was trained on **FaceAVCeleb
face-swap frames** — short video clips where a real face is swapped onto another person.
Its validation accuracy on that task is roughly **~63%**.

The model performs poorly on **AI-generated portrait photos** produced by diffusion models
(Stable Diffusion, Midjourney, DALL-E, Flux) or GANs, because those images look nothing
like the face-swap artifacts it was trained to detect. As a result, it frequently classifies
AI-generated images as `REAL`.

**CIFAKE** is a benchmark of 140,000 labelled images — real CIFAR-10 photos versus Stable
Diffusion synthetic equivalents. Retraining on CIFAKE teaches the model to distinguish
photographic authenticity from AI-generated textures, boosting expected accuracy to
**~85–92%** on that domain.

---

## What Is CIFAKE?

CIFAKE (Bird & Chowdhury, 2023) is a benchmark dataset:

| Split | REAL | FAKE | Total |
|-------|------|------|-------|
| Train | 60,000 | 60,000 | 120,000 |
| Test  | 10,000 | 10,000 | 20,000 |

- **REAL** — photographs from CIFAR-10 (32×32, resized to 160×160 during training)
- **FAKE** — Stable Diffusion v1.4 synthetic equivalents of every CIFAR-10 class

Source: [Kaggle — birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

---

## Label Convention

| Class | Integer | Meaning |
|-------|---------|---------|
| FAKE  | **0**   | AI-generated / manipulated image |
| REAL  | **1**   | Authentic photograph |

This convention is hardcoded in `models/image_model.py`, `api/pipeline.py`, and
`train/cifake_dataset.py`. **Do not change it.**

---

## Prerequisites

```bash
pip install kaggle torch torchvision
```

Set up Kaggle credentials (one of the following):

**Option A — Environment variables:**
```bash
# Windows PowerShell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_api_key"
```
```bash
# Linux / macOS
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Option B — kaggle.json file:**  
Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS) or
`C:\Users\<you>\.kaggle\kaggle.json` (Windows).

Get your API key at <https://www.kaggle.com/settings/account>.

---

## Step-by-Step Instructions

### Step 1 — Download CIFAKE

```bash
cd c:\Projects\Multimodal_AV_fraud_and_deepfake
python train/download_cifake.py
```

Expected output:
```
Downloading CIFAKE dataset from Kaggle...
  Dataset : birdy654/cifake-real-and-ai-generated-synthetic-images
  Target  : <project_root>/data/cifake

Verifying downloaded structure...
  train/REAL: 60000 images
  train/FAKE: 60000 images
  test/REAL:  10000 images
  test/FAKE:  10000 images

CIFAKE download complete.
```

Dataset lands at `data/cifake/`.  
If data is already present, use `--skip-download` to just print counts:

```bash
python train/download_cifake.py --skip-download
```

---

### Step 2 — Train the Image Classifier

```bash
cd c:\Projects\Multimodal_AV_fraud_and_deepfake
python train/train_cifake.py
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce to 8 on CPU; increase to 64 with a strong GPU) |
| `--learning_rate` | 1e-4 | Head learning rate (backbone gets lr × 0.1) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--num_workers` | 4 | DataLoader workers (use 0 on Windows if spawn errors occur) |
| `--pretrained` | True | Warm-start from existing `efficientnet_b0_image_best.pth` |
| `--data_root` | `data/cifake` | Path to the CIFAKE root folder |

**Fixed training configuration:**
- Image size: **160 × 160** (CIFAKE images are 32×32 native, resized to match the model)
- Normalize mean: `[0.485, 0.456, 0.406]`
- Normalize std: `[0.229, 0.224, 0.225]`
- Training augmentations: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`,
  `RandomAffine(translate=(0.08,0.08), scale=(0.92,1.08))`,
  `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- Backbone freeze: all `model.backbone.features[:-4]` frozen; last 4 blocks + classifier trained
- Optimizer: AdamW — backbone `lr=1e-5`, head `lr=1e-4`, `weight_decay=1e-4`
- Scheduler: CosineAnnealingLR(`T_max=epochs`, `eta_min=1e-6`)
- Gradient clipping: `max_norm=1.0`
- Loss: CrossEntropyLoss with balanced class weights
- Early stopping: patience = 5 epochs on val accuracy

Checkpoints are saved to:
```
checkpoints/image_deepfake/efficientnet_b0_cifake_best.pth   ← best val accuracy
checkpoints/image_deepfake/efficientnet_b0_cifake_final.pth  ← final epoch
```

---

### Step 3 — Swap the Checkpoint in the API

Edit `api/pipeline.py` and change:

```python
IMAGE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "image_deepfake" / "efficientnet_b0_image_best.pth"
```

to:

```python
IMAGE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "image_deepfake" / "efficientnet_b0_cifake_best.pth"
```

Then restart the API server. On next startup you should see:
```
- Pure Image Deepfake Classifier (EfficientNet-B0) loaded.
```

Alternatively, copy the new checkpoint over the old one so no code change is needed:
```bash
# Windows
copy "checkpoints\image_deepfake\efficientnet_b0_cifake_best.pth" "checkpoints\image_deepfake\efficientnet_b0_image_best.pth"
```

---

## Expected Training Time

| Hardware | Approx. time per epoch | Total (15 epochs) |
|----------|------------------------|-------------------|
| CPU only (4-core) | ~90–120 min | ~22–30 hours |
| GPU (RTX 3060) | ~4–6 min | ~60–90 min |
| GPU (RTX 4090) | ~1–2 min | ~15–30 min |

> CIFAKE images are 32×32 so individual forward passes are fast; the bottleneck is
> DataLoader throughput and the large dataset size (120,000 train images).

---

## Expected Accuracy

| Checkpoint | Training data | Val accuracy |
|------------|---------------|-------------|
| `efficientnet_b0_image_best.pth` | FaceAVCeleb face-swaps | ~63% |
| `efficientnet_b0_cifake_best.pth` | CIFAKE (AI-gen vs real) | ~85–92% |

Literature reports 92–95% with full fine-tuning on CIFAKE. With partial backbone freeze
(last 4 blocks only), 88–92% is the typical range.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: CIFAKE split directory not found` | Run `python train/download_cifake.py` first |
| `kaggle CLI not found` | Run `pip install kaggle` |
| `Kaggle credentials not found` | Set env vars or place `kaggle.json` at `~/.kaggle/` |
| CUDA out of memory | Reduce `--batch_size` to 8; or run on CPU |
| Model still classifies AI images as REAL after swap | Confirm checkpoint `val_accuracy` ≥ 0.85; try `--epochs 20` |
| `Image model checkpoint not loaded` in API logs | Verify the checkpoint path in `api/pipeline.py` points to the new file |
| DataLoader spawn errors on Windows | Add `--num_workers 0` |

---

## File Reference

| File | Purpose |
|------|---------|
| `train/download_cifake.py` | Downloads CIFAKE via Kaggle CLI |
| `train/cifake_dataset.py` | PyTorch Dataset wrapping CIFAKE folder layout |
| `train/train_cifake.py` | Full retraining script |
| `api/pipeline.py` `IMAGE_CHECKPOINT` | Path the API reads at startup |
| `models/image_model.py` | `ImageDeepfakeModel` architecture (EfficientNet-B0) |
