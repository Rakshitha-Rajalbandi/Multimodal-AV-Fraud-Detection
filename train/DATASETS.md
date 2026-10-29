# Dataset Reference — Multimodal AV Fraud & Deepfake Detector

This document covers all supported image datasets: their download commands,
expected directory structures, approximate sizes, and the shared label convention.

---

## Label Convention

| Label | Meaning        | Description                             |
|-------|----------------|-----------------------------------------|
| **0** | **FAKE**       | AI-generated or manipulated image       |
| **1** | **REAL**       | Authentic, unmodified photograph        |

This convention is fixed by `models/image_model.py` (`ImageDeepfakeModel`) and
`api/pipeline.py`. All dataset classes follow it.

---

## CIFAKE

- **Size:** ~120 K images (60 K train + 20 K test each class)
- **Source:** Kaggle — `birdy654/cifake-real-and-ai-generated-synthetic-images`
- **Generator coverage:** Stable Diffusion v1.4 (FAKE); CIFAR-10 photographs (REAL)

### Download

```bash
python train/download_cifake.py
# Already downloaded:
python train/download_cifake.py --skip-download
```

### Directory structure after download

```
data/cifake/
  train/
    REAL/   60 000 × 32×32 jpg  (label 1)
    FAKE/   60 000 × 32×32 jpg  (label 0)
  test/
    REAL/   10 000 × 32×32 jpg  (label 1)
    FAKE/   10 000 × 32×32 jpg  (label 0)
```

### PyTorch Dataset class

```python
from train.cifake_dataset import CIFAKEDataset
ds = CIFAKEDataset(root="data/cifake", split="train", training=True)
```

---

## ArtiFact

- **Size:** ~2.5 M images
- **Source:** Kaggle — `awsaf49/artifact-dataset`
- **Generator coverage:** real photographs PLUS Midjourney, DALL-E, Stable Diffusion, VQDM,
  BigGAN, ADM, GLIDE, Wukong, and other generators.

### Download

```bash
# Requires: pip install kaggle + ~/.kaggle/kaggle.json
python train/download_artifact.py
# Already downloaded:
python train/download_artifact.py --skip-download
```

### Directory structure after download

```
data/artifact/
  real/              → label 1  (REAL — authentic photographs)
  midjourney/        → label 0  (FAKE)
  dalle/             → label 0  (FAKE)
  stablediffusion/   → label 0  (FAKE)
  vqdm/              → label 0  (FAKE)
  biggan/            → label 0  (FAKE)
  adm/               → label 0  (FAKE)
  glide/             → label 0  (FAKE)
  wukong/            → label 0  (FAKE)
  (other AI subdir)  → label 0  (FAKE)
```

_Any subdirectory not named `real` is treated as FAKE._

### PyTorch Dataset class

```python
from train.artifact_dataset import ArtifactDataset
ds = ArtifactDataset(root="data/artifact", split="train", training=True)
```

---

## GenImage

- **Size:** ~1.3 M images
- **Source:** Hugging Face — `zhwang/GenImage`
- **Generator coverage:** BigGAN, VQDM, ADM, GLIDE, Midjourney, Stable Diffusion v4/v5,
  Wukong (FAKE); ImageNet photographs via GLIDE real split (REAL).

### Download

```bash
# Requires: pip install huggingface_hub
python train/download_genimage.py
# Already downloaded:
python train/download_genimage.py --skip-download
```

### Directory structure after download

```
data/genimage/
  imagenet_glide/            → label 1  (REAL — authentic ImageNet photos)
  imagenet_ai_0419_biggan/   → label 0  (FAKE)
  imagenet_ai_0419_vqdm/     → label 0  (FAKE)
  imagenet_ai_0508_adm/      → label 0  (FAKE)
  imagenet_ai_0508_glide/    → label 0  (FAKE)
  imagenet_ai_0419_midjourney/ → label 0  (FAKE)
  imagenet_ai_0419_sdv4/     → label 0  (FAKE)
  imagenet_ai_0419_sdv5/     → label 0  (FAKE)
  imagenet_ai_0424_wukong/   → label 0  (FAKE)
```

_Any subdirectory whose name contains `imagenet_glide` is REAL; all others are FAKE._

### PyTorch Dataset class

```python
from train.genimage_dataset import GenImageDataset
ds = GenImageDataset(root="data/genimage", split="train", training=True)
```

---

## Shared Transform Configuration

All dataset classes use identical transforms (matching `train/cifake_dataset.py`):

| Setting         | Value                                              |
|-----------------|----------------------------------------------------|
| `image_size`    | 160×160 pixels                                     |
| Normalize mean  | `[0.485, 0.456, 0.406]` (ImageNet)                 |
| Normalize std   | `[0.229, 0.224, 0.225]` (ImageNet)                 |

**Training augmentation** (applied when `training=True`):
1. `Resize(160, 160)`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomRotation(degrees=15)`
4. `RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08))`
5. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
6. `ToTensor()`
7. `Normalize(mean, std)`

**Val / Test transforms** (applied when `training=False`):
1. `Resize(160, 160)`
2. `ToTensor()`
3. `Normalize(mean, std)`

---

## Deterministic Train/Val/Test Splits (ArtiFact and GenImage)

For ArtiFact and GenImage, splits are computed by **sorted file list + index slicing**
(no random shuffle), ensuring reproducibility across runs:

| Split | Proportion |
|-------|------------|
| train | 80%        |
| val   | 10%        |
| test  | 10%        |

CIFAKE uses the pre-defined `train/` and `test/` folder split provided by the dataset.

---

## Combined Training

To train on all three datasets concatenated:

```python
from torch.utils.data import ConcatDataset
from train.cifake_dataset import CIFAKEDataset
from train.artifact_dataset import ArtifactDataset
from train.genimage_dataset import GenImageDataset

train_ds = ConcatDataset([
    CIFAKEDataset(split="train", training=True),
    ArtifactDataset(split="train", training=True),
    GenImageDataset(split="train", training=True),
])
```
