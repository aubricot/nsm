"""
Evaluation metrics aligned with Project Proposal V2 requirements.

Computes:
  - Instance Accuracy
  - Average Class Accuracy (balanced)
  - Precision  (macro, weighted, per-class)
  - Recall     (macro, weighted, per-class)
  - F1         (macro, weighted, per-class)
  - Top-K Accuracy (when probabilities are available)
  - Per-class support counts
  - Confusion matrices at hierarchical taxonomy levels
  - Regression metrics for continuous spine position (MAE, RMSE, R², Spearman)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr
from typing import Dict, List, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_probs: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Calculates the full set of classification metrics required by the
    Project Proposal V2:

      • Instance Accuracy
      • Average Class Accuracy (balanced)
      • Precision  (macro + weighted)
      • Recall     (macro + weighted)
      • F1         (macro + weighted)
      • Top-K Accuracy (if probability vectors provided)
      • Per-class breakdown (precision / recall / f1 / support for each label)

    Parameters
    ----------
    y_true : list of str
        Ground truth labels.
    y_pred : list of str
        Predicted labels.
    y_probs : np.ndarray, optional
        Probability matrix of shape (n_samples, n_classes).
    classes : np.ndarray, optional
        Ordered class labels corresponding to columns of *y_probs*.
    top_k : int
        K for top-K accuracy (only computed when y_probs is provided).

    Returns
    -------
    dict
        Keys are metric names; values are floats or nested dicts for
        per-class metrics.
    """
    if len(y_true) == 0:
        return {}

    metrics: Dict[str, Any] = {}

    # ---- Summary statistics ----
    metrics["instance_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["average_class_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Macro-averaged
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = prec_macro
    metrics["recall_macro"] = rec_macro
    metrics["f1_macro"] = f1_macro

    # Weighted-averaged (accounts for class imbalance)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["precision_weighted"] = prec_w
    metrics["recall_weighted"] = rec_w
    metrics["f1_weighted"] = f1_w

    # ---- Per-class breakdown ----
    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, average=None, zero_division=0
    )
    per_class = {}
    for i, label in enumerate(unique_labels):
        per_class[label] = {
            "precision": float(prec_pc[i]),
            "recall": float(rec_pc[i]),
            "f1": float(f1_pc[i]),
            "support": int(sup_pc[i]),
        }
    metrics["per_class"] = per_class

    # ---- Top-K Accuracy ----
    if y_probs is not None and classes is not None and top_k > 1:
        n_classes = len(classes)
        k = min(top_k, n_classes)
        top_k_indices = np.argsort(y_probs, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            true_idx = np.where(classes == true_label)[0]
            if len(true_idx) > 0 and true_idx[0] in top_k_indices[i]:
                correct += 1
        metrics[f"top_{top_k}_accuracy"] = correct / len(y_true)

    # Total samples
    metrics["n_samples"] = len(y_true)

    return metrics


def metrics_to_dataframe(metrics: Dict[str, Any], classifier_name: str = "") -> pd.DataFrame:
    """
    Converts the dict returned by :func:`calculate_metrics` into a tidy
    DataFrame suitable for CSV export.

    Summary metrics are one row; per-class metrics are additional rows.
    """
    rows = []

    # Summary row
    summary = {
        "classifier": classifier_name,
        "level": "summary",
        "class": "ALL",
    }
    for k, v in metrics.items():
        if k not in ("per_class",):
            summary[k] = v
    rows.append(summary)

    # Per-class rows
    if "per_class" in metrics:
        for label, pc_metrics in metrics["per_class"].items():
            row = {
                "classifier": classifier_name,
                "level": "per_class",
                "class": label,
            }
            row.update(pc_metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hierarchical confusion matrices
# ---------------------------------------------------------------------------

def generate_hierarchical_confusion_matrices(
    y_true_dicts: List[Dict[str, str]],
    y_pred_dicts: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Generates confusion matrices at different taxonomic levels
    (family, genus, species).

    Parameters
    ----------
    y_true_dicts, y_pred_dicts : list of dict
        Each dict should contain keys 'family', 'genus', 'species'.

    Returns
    -------
    dict
        Keys are level names; values are dicts with 'matrix' and 'labels'.
    """
    levels = ["family", "genus", "species"]
    matrices = {}

    for level in levels:
        y_true_level = [d.get(level, "unknown") for d in y_true_dicts]
        y_pred_level = [d.get(level, "unknown") for d in y_pred_dicts]

        labels = sorted(set(y_true_level + y_pred_level))
        cm = confusion_matrix(y_true_level, y_pred_level, labels=labels)
        matrices[level] = {"matrix": cm, "labels": labels}

    return matrices


def generate_position_confusion_matrix(
    y_true_positions: List[str],
    y_pred_positions: List[str],
) -> Dict[str, Any]:
    """
    Generates a confusion matrix specifically for spinal position prediction.
    """
    labels = sorted(set(y_true_positions + y_pred_positions))
    cm = confusion_matrix(y_true_positions, y_pred_positions, labels=labels)
    return {"matrix": cm, "labels": labels}


# ---------------------------------------------------------------------------
# Regression metrics for continuous spine position
# ---------------------------------------------------------------------------

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates regression metrics for continuous position prediction.

    Parameters
    ----------
    y_true : array-like
        Ground truth continuous positions (0-1 range).
    y_pred : array-like
        Predicted continuous positions.

    Returns
    -------
    dict
        Keys: mae, rmse, r2, spearman_rho, spearman_pvalue, n_samples
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    rho, pval = spearmanr(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_samples": int(len(y_true)),
    }
