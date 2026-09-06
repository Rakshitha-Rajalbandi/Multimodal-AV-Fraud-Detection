import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_equal_error_rate(y_true, y_prob):
    """
    Calculate Equal Error Rate (EER) given true binary labels and predicted probabilities.
    """
    if y_prob is None or len(np.unique(y_true)) < 2:
        return 0.0

    thresholds = np.linspace(0, 1, 1000)
    fnr_list, fpr_list = [], []

    for t in thresholds:
        preds = (np.array(y_prob) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        fnr_list.append(fnr)
        fpr_list.append(fpr)

    fnr_arr = np.array(fnr_list)
    fpr_arr = np.array(fpr_list)
    diff = np.abs(fnr_arr - fpr_arr)
    min_idx = np.argmin(diff)
    eer = float((fnr_arr[min_idx] + fpr_arr[min_idx]) / 2.0)
    return eer


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes classification metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, EER.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = 0.0
    eer = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            eer = calculate_equal_error_rate(y_true, y_prob)
        except Exception:
            pass

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4)
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """
    Generate and save visual confusion matrix plot using Matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['REAL', 'FAKE'])
    ax.set_yticklabels(['REAL', 'FAKE'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return cm


def print_metrics_summary(metrics, title="EVALUATION METRICS"):
    """
    Print formatted metrics table.
    """
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k.replace('_', ' ').title():<20}: {v:.4f}")
    print("=" * 50 + "\n")
