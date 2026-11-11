"""
=============================================================================
EmotiSync - Black & White Academic Line Graph: Algorithm Comparison
AGB1303 - AI Problem Solving Techniques
Team: Vishwajith, Wilson, Vadivazhagi, Viseka
=============================================================================
Generates a strictly monochrome (Black & White) publication-grade line graph
comparing all algorithms used across EmotiSync:
  Panel A: Multi-metric Performance Comparison (Accuracy, F1, Precision, Recall)
  Panel B: Training Convergence Trajectory & Multimodal Fusion Uplift
"""

import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_PNG = REPORTS_DIR / "algorithm_comparison_bw_line_graph.png"

def main():
    # Load genuine training histories from reports/
    with open(REPORTS_DIR / "facial_training_history.json", "r", encoding="utf-8") as f:
        face_hist = json.load(f)
    with open(REPORTS_DIR / "speech_training_history.json", "r", encoding="utf-8") as f:
        speech_hist = json.load(f)

    # Strictly monochrome IEEE / Academic styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.1,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # =========================================================================
    # PANEL A: Algorithm Comparison Across Evaluation Metrics
    # Metrics sourced directly from reports/evaluation_report.md & Ch. 9 PDF
    # =========================================================================
    algorithms = [
        "Baseline\nFacial CNN",
        "YOLOv8 +\nVGG-FER CNN",
        "82-D MFCC\n+ BiLSTM",
        "Dual-Modal\nFusion (70/30)",
        "Tri-Modal\nBayesian Fusion",
        "VADER + Lexicon\n(Semantic NLP)"
    ]
    x = np.arange(len(algorithms))

    test_acc = [67.66, 72.94, 94.69, 95.62, 96.20, 98.40]
    weighted_f1 = [67.29, 72.45, 94.66, 95.60, 95.85, 98.35]

    # Precision & Recall reported for the 5 core production benchmarks (Ch. 9)
    x_pr = np.arange(1, len(algorithms))
    precision = [73.00, 95.00, 96.00, 96.00, 98.00]
    recall = [73.00, 95.00, 96.00, 96.00, 98.00]

    # Plot lines with distinct B&W line styles & geometric markers
    ax1.plot(
        x, test_acc,
        color="#000000", linestyle="-", linewidth=2.2,
        marker="o", markersize=7.5, markerfacecolor="#000000", markeredgecolor="#000000",
        label="Test Accuracy (%)", zorder=5
    )
    ax1.plot(
        x, weighted_f1,
        color="#333333", linestyle="--", linewidth=1.8,
        marker="s", markersize=7.0, markerfacecolor="#FFFFFF", markeredgecolor="#000000", markeredgewidth=1.4,
        label="Weighted F1-Score (%)", zorder=4
    )
    ax1.plot(
        x_pr, precision,
        color="#555555", linestyle=":", linewidth=1.7,
        marker="^", markersize=6.8, markerfacecolor="#888888", markeredgecolor="#000000", markeredgewidth=1.0,
        label="Precision (%)", zorder=3
    )
    ax1.plot(
        x_pr, recall,
        color="#777777", linestyle="-.", linewidth=1.5,
        marker="D", markersize=5.8, markerfacecolor="#FFFFFF", markeredgecolor="#333333", markeredgewidth=1.2,
        label="Recall (%)", zorder=2
    )

    # Annotate Accuracy & F1 values cleanly without collisions
    for idx, (acc_val, f1_val) in enumerate(zip(test_acc, weighted_f1)):
        acc_offset = (-18, 10) if idx == 1 else (0, 9)
        f1_offset = (18, -14) if idx == 2 else (0, -14)
        ax1.annotate(
            f"{acc_val:.2f}%",
            xy=(idx, acc_val),
            xytext=acc_offset,
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color="#000000"
        )
        ax1.annotate(
            f"{f1_val:.2f}%",
            xy=(idx, f1_val),
            xytext=f1_offset,
            textcoords="offset points",
            ha="center", va="top",
            fontsize=7.6, color="#333333"
        )

    ax1.set_title(
        "(a) Quantitative Comparison of EmotiSync Algorithms",
        fontsize=11.5, fontweight="bold", pad=12, color="#000000"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=8.4, fontweight="bold")
    ax1.set_ylabel("Performance Score (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(60, 105)
    ax1.grid(True, linestyle=":", linewidth=0.7, color="#BDBDBD", alpha=0.85)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="lower right", frameon=True, facecolor="#FFFFFF",
        edgecolor="#000000", fontsize=8.2, framealpha=1.0
    )

    # =========================================================================
    # PANEL B: Validation Accuracy Over Training Epochs & Fusion Ceiling
    # Metrics sourced directly from facial_training_history.json & speech_training_history.json
    # =========================================================================
    face_epochs = np.arange(1, len(face_hist["val_acc"]) + 1)
    speech_epochs = np.arange(1, len(speech_hist["val_acc"]) + 1)

    ax2.plot(
        speech_epochs, speech_hist["val_acc"],
        color="#000000", linestyle="-", linewidth=2.2,
        marker="o", markersize=6.5, markerfacecolor="#000000", markeredgecolor="#000000",
        label="Speech BiLSTM (82-D MFCC) Val Acc"
    )
    ax2.plot(
        face_epochs, face_hist["val_acc"],
        color="#222222", linestyle="--", linewidth=2.0,
        marker="s", markersize=6.2, markerfacecolor="#FFFFFF", markeredgecolor="#000000", markeredgewidth=1.4,
        label="Facial VGG-FER CNN Val Acc"
    )

    # Multimodal Fusion Reference Baselines
    ax2.axhline(
        y=95.62, color="#444444", linestyle="-.", linewidth=1.5,
        label="Dual-Modal Fusion (70/30): 95.62%"
    )
    ax2.axhline(
        y=96.20, color="#000000", linestyle=":", linewidth=1.7,
        label="Tri-Modal Bayesian Fusion Peak: 96.20%"
    )

    # Peak epoch annotations
    ax2.annotate(
        f"Peak: {speech_hist['val_acc'][-1]:.2f}%",
        xy=(speech_epochs[-1], speech_hist["val_acc"][-1]),
        xytext=(-12, -16), textcoords="offset points",
        ha="right", fontsize=8.0, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#000000", lw=0.9)
    )
    ax2.annotate(
        f"Epoch 8: {face_hist['val_acc'][-1]:.2f}%",
        xy=(face_epochs[-1], face_hist["val_acc"][-1]),
        xytext=(12, -14), textcoords="offset points",
        ha="left", fontsize=8.0, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#000000", lw=0.9)
    )

    ax2.set_title(
        "(b) Neural Learning Curves vs. Multimodal Fusion Ceiling",
        fontsize=11.5, fontweight="bold", pad=12, color="#000000"
    )
    ax2.set_xlabel("Training Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Validation / System Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(np.arange(1, 13))
    ax2.set_xlim(0.5, 12.6)
    ax2.set_ylim(60, 102)
    ax2.grid(True, linestyle=":", linewidth=0.7, color="#BDBDBD", alpha=0.85)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        loc="center right", bbox_to_anchor=(0.99, 0.42),
        frameon=True, facecolor="#FFFFFF", edgecolor="#000000",
        fontsize=8.0, framealpha=1.0
    )

    fig.suptitle(
        "EmotiSync Algorithm Comparison — Accuracy, F1-Score, Precision, Recall & Convergence (Monochrome B&W)",
        fontsize=13, fontweight="bold", y=0.99, color="#000000"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved monochrome B&W algorithm comparison line graph to: {OUTPUT_PNG}")

    # Copy to conversation artifact directory for direct inline display
    artifact_dir = Path(r"C:\Users\HP\.gemini\antigravity\brain\50fa836f-6247-4cd3-8ec5-537075ea90c8")
    if artifact_dir.exists():
        artifact_png = artifact_dir / "algorithm_comparison_bw_line_graph.png"
        shutil.copy2(OUTPUT_PNG, artifact_png)
        print(f"Copied graph to artifact directory: {artifact_png}")


if __name__ == "__main__":
    main()
