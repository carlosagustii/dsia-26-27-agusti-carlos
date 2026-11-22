"""Métricas de evaluación de probabilidades y curva de calibración."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: funciona en CI y servidores

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402


def evaluate(y_true, y_prob) -> dict:
    """Log loss (métrica principal), Brier y ROC-AUC, redondeadas a 4 decimales.

    Se acotan las probabilidades para que un 0 o 1 exacto no dé log loss infinito.
    """
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def plot_calibration(y_true, probs_by_name: dict, output_path: Path, n_bins: int = 10) -> Path:
    """Dibuja la curva de calibración de cada modelo frente a la diagonal ideal."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Calibración perfecta")
    for name, probs in probs_by_name.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("xG predicho (media del bin)")
    ax.set_ylabel("Proporción real de goles")
    ax.set_title("Curva de calibración · LaLiga 2020-21 sin penaltis")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
