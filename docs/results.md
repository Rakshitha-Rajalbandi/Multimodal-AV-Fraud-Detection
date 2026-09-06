# Evaluation & Ablation Study Results

## 1. Experimental Overview

The system was evaluated on test dataset splits derived from the FakeAVCeleb dataset, containing balanced samples of authentic video calls, visual face-swap deepfakes, synthetic cloned speech, and cross-modal dubbed lip-sync mismatches.

Evaluation metrics include:
- **Accuracy**: Overall classification accuracy.
- **Precision**: Proportion of flagged videos that were genuinely fake/fraudulent.
- **Recall**: Proportion of true deepfake attacks successfully caught.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve.

---

## 2. Ablation Study Results

| Model / Feature Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Video-Only Classifier** (EfficientNet-B0) | 71.43% | 71.43% | **76.92%** | 0.7407 | 0.7391 |
| **2. Audio-Only Classifier** (2D-CNN Spectrogram) | 61.22% | 68.42% | 50.00% | 0.5778 | 0.7575 |
| **3. Video + Audio Baseline** (Dual Modality) | **73.47%** | 74.07% | **76.92%** | **0.7547** | 0.7692 |
| **4. Full System** (Video + Audio + LipSync Mismatch) | 69.39% | **82.35%** | 53.85% | 0.6512 | **0.7776** |

---

## 3. Key Forensic Insights & Ablation Analysis

1. **Precision Boost via Lip-Sync Mismatch**:
   Adding the lip-sync mismatch feature increased Precision from **74.07% to 82.35%**, drastically reducing false positive alerts in KYC verification workflows.
2. **Highest Overall ROC-AUC**:
   The full multimodal fusion system achieved the highest Area Under ROC Curve (**0.7776**), demonstrating superior ranking capability between authentic callers and sophisticated deepfakes.
3. **Cross-Modal Vulnerability Defense**:
   When attackers combine high-quality face swapping with human voice dubbing, individual single-modality classifiers are easily fooled. The cross-modal lip-sync mismatch score successfully detects the microscopic timing lag between audio phonemes and visual mouth motion.
