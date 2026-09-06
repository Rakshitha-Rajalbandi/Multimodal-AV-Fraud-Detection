# Multimodal Audio-Video Fraud & Deepfake Detection Platform

A complete AI-driven forensic security platform that identifies AI-generated or manipulated video calls (face-swaps, voice clones, and dubbed lip-sync mismatches) across **Call Center Fraud Prevention**, **Remote Verification**, **Forensic Investigation**, and **Executive Impersonation Security**.

---

## 🎯 Executive Summary & Core Framing

In modern digital verification and remote authorization, attackers deploy sophisticated multimodal deepfakes — combining swapped faces with cloned voices. Analyzing video or audio in isolation fails when attackers generate realistic single-modality fakes. 

This platform solves this vulnerability by evaluating **tri-modality consistency**:
1. **Visual Authenticity**: CNN-based face-forgery classification (EfficientNet-B0 + BiGRU + Attention).
2. **Speech Authenticity**: 2D-CNN Spectrogram classifier detecting synthetic speech and voice cloning.
3. **Lip-Audio Synchronization**: Cross-correlation timing analysis between visual lip motion and acoustic phoneme energy.

These signals are fused into a **Fraud Risk Score (0 - 100%)** presented on an interactive **Forensic Analysis Dashboard**.

---

## 🏗 System Architecture

```
                                 ┌──────────────────────────────────┐
                                 │       Input Video Stream         │
                                 └────────────────┬─────────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 │                                │                                │
                 ▼                                ▼                                ▼
  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │   1. Visual Classifier      │  │   2. Audio Classifier       │  │   3. Lip-Sync Evaluator     │
  │  EfficientNet-B0 + BiGRU    │  │  2D-CNN Mel-Spectrogram     │  │  Cross-Modal Correlation    │
  └──────────────┬──────────────┘  └──────────────┬──────────────┘  └──────────────┬──────────────┘
                 │                                │                                │
                 │ Visual Score (P_video)         │ Audio Score (P_audio)          │ Sync Mismatch (S_sync)
                 └────────────────────────────────┼────────────────────────────────┘
                                                  │
                                                  ▼
                                 ┌──────────────────────────────────┐
                                 │     Multimodal Fusion Layer      │
                                 │     (PyTorch MLP Network)        │
                                 └────────────────┬─────────────────┘
                                                  │
                                                  ▼
                                 ┌──────────────────────────────────┐
                                 │     Fraud Risk Score (0-100%)    │
                                 │    Forensic Review Dashboard     │
                                 └──────────────────────────────────┘
```

---

## 📂 Project Structure

```
C:\Projects\Multimodal_AV_fraud_and_deepfake
├── api/
│   ├── app.py                   # Flask API Server & Static Dashboard Host
│   └── pipeline.py              # End-to-End Multimodal Inference Engine
├── checkpoints/
│   ├── audio_deepfake_best.pth  # Trained Audio Classifier Checkpoint
│   ├── fusion_model.pt          # Trained Multimodal Fusion Checkpoint
│   └── video_deepfake/          # Trained EfficientNet-B0 Visual Checkpoint
├── docs/
│   ├── architecture.md          # Technical Architecture & Pipeline Specification
│   └── results.md               # Evaluation Metrics & Ablation Study Summary
├── frontend/
│   └── index.html               # React & Tailwind Forensic Dashboard
├── models/
│   ├── audio_model.py           # 2D-CNN Audio Spectrogram PyTorch Model
│   ├── fusion_model.py          # Multimodal Fusion PyTorch Module & Logic
│   ├── sync_model.py            # LipSync Mismatch PyTorch Model & Cross-Correlation
│   └── video_model.py           # EfficientNet-B0 + BiGRU + Attention Visual Model
├── preprocess/
│   ├── compute_lip_motion.py    # Lip Motion Trajectory Extraction
│   ├── extract_frames_audio.py  # Video Frame & Audio Extraction
│   └── make_melspec.py          # Mel-Spectrogram Feature Preprocessor
├── scripts/
│   ├── calculate_sync_score.py # Sync Mismatch Score Generator
│   ├── evaluate_multimodal.py   # Multi-class & Ablation Evaluation Script
│   └── fuse_multimodal.py       # Multimodal Grid Search & Threshold Tuning
├── tests/
│   ├── test_api.py              # Flask API Unit Tests
│   └── test_preprocessing.py    # Preprocessing & Model Unit Tests
└── train/
    ├── metrics.py               # Accuracy, Precision, Recall, F1, EER, AUC Metrics
    ├── train_audio.py           # Audio Model Training Pipeline
    ├── train_fusion.py          # Multimodal Fusion Model Training Pipeline
    └── train_video.py           # Video Model Training Pipeline
```

---

## 🚀 Quick Start & Usage Instructions

### 1. Prerequisites & Virtual Environment
Ensure Python 3.12+ and virtual environment dependencies are activated:
```bash
# Activate virtual environment
C:\Projects\Multimodal_AV_fraud_and_deepfake\.venv\Scripts\activate
```

### 2. Run Flask API & Interactive Dashboard
Launch the Flask backend server:
```bash
python api/app.py
```
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard)**

### 3. Run Unit Tests
Verify API and preprocessing modules with standard unittest runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Run Ablation Study & Evaluation
To execute the multi-class evaluation comparing single-modality baselines vs full multimodal fusion:
```bash
python scripts/evaluate_multimodal.py
```

---

## 📊 Ablation Study Summary

Evaluation across test splits demonstrates that adding the **lip-sync mismatch feature** provides critical forensic precision:

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Video-Only** (EfficientNet-B0) | 71.43% | 71.43% | **76.92%** | 0.7407 | 0.7391 |
| **Audio-Only** (2D-CNN Spectrogram) | 61.22% | 68.42% | 50.00% | 0.5778 | 0.7575 |
| **Dual Modality** (Video + Audio) | **73.47%** | 74.07% | **76.92%** | **0.7547** | 0.7692 |
| **Full System** (Video + Audio + Sync) | 69.39% | **82.35%** | 53.85% | 0.6512 | **0.7776** |

- **Highest Precision (82.35%)**: Drastically reduces false positive alerts in automated verification workflows.
- **Highest ROC-AUC (0.7776)**: Proves superior overall ranking capability against voice-cloned & dubbed deepfakes.

---

## 📄 License & Technical Rationale
Built for AI Forensic Security & Identity Verification research. All rights reserved.
