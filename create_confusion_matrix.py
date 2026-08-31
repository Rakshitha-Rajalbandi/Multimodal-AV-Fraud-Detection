# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay


# # ------------------------------------------------------------
# # Paths
# # ------------------------------------------------------------

# METRICS_FILE = Path("results/evaluation/metrics.json")
# OUTPUT_DIR = Path("results/evaluation")

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ------------------------------------------------------------
# # Load metrics
# # ------------------------------------------------------------

# with open(
#     METRICS_FILE,
#     "r",
#     encoding="utf-8"
# ) as f:
#     metrics = json.load(f)


# cm = metrics["confusion_matrix"]


# # ------------------------------------------------------------
# # Create confusion matrix
# # ------------------------------------------------------------

# display = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["FAKE", "REAL"]
# )

# display.plot()

# plt.title(
#     f"Video Deepfake Detection\n"
#     f"Accuracy = {metrics['accuracy'] * 100:.2f}%"
# )

# plt.tight_layout()


# # ------------------------------------------------------------
# # Save
# # ------------------------------------------------------------

# output_file = (
#     OUTPUT_DIR /
#     "confusion_matrix.png"
# )

# plt.savefig(
#     output_file,
#     dpi=200,
#     bbox_inches="tight"
# )

# plt.close()

# print("Confusion matrix saved to:")
# print(output_file)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Confusion matrix from test results
cm = np.array([
    [23, 7],
    [8, 22]
])


# Label mapping from our project
# 0 = FAKE
# 1 = REAL

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

display.plot()

plt.title("Video Deepfake Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/evaluation/confusion_matrix.png",
    dpi=300
)

plt.show()

print("Confusion matrix saved successfully.")