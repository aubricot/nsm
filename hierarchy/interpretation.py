"""
Logistic Regression interpretation for latent space classifiers.

Extracts coefficients and computes Wald-test p-values from a fitted
sklearn LogisticRegression (optionally inside a Pipeline and/or
MultiOutputClassifier), then generates visualizations showing which
latent dimensions drive each class prediction.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================
# Extraction helpers
# ======================================================================

def _unwrap_lr(model) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """Unwrap a LogisticRegression from Pipeline / MultiOutputClassifier layers.

    Returns (lr_model, scaler_or_None).
    """
    if isinstance(model, LogisticRegression):
        return model, None

    if isinstance(model, Pipeline):
        scaler = None
        lr = None
        for step_name, step in model.steps:
            if isinstance(step, StandardScaler):
                scaler = step
            if isinstance(step, LogisticRegression):
                lr = step
        if lr is None:
            raise ValueError("Pipeline does not contain a LogisticRegression step")
        return lr, scaler

    raise TypeError(f"Cannot unwrap LogisticRegression from {type(model)}")


def extract_position_lr(
    multi_output_model: MultiOutputClassifier,
    target_index: int = 2,
) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """Extract the position LogisticRegression from a MultiOutputClassifier.

    Parameters
    ----------
    multi_output_model : fitted MultiOutputClassifier
    target_index : index of the position target (default 2: species=0, genus=1, position=2)
    """
    estimator = multi_output_model.estimators_[target_index]
    return _unwrap_lr(estimator)


# ======================================================================
# Coefficient analysis
# ======================================================================

def get_coefficients(
    lr: LogisticRegression,
    scaler: Optional[StandardScaler] = None,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract coefficient matrix as a DataFrame.

    Returns coefficients in both scaled and original feature space.
    Rows = classes, columns = features.

    Parameters
    ----------
    lr : fitted LogisticRegression
    scaler : fitted StandardScaler (if LR was trained on scaled data)
    feature_names : names for each feature dimension (default: latent_0, latent_1, ...)
    """
    coef_scaled = lr.coef_  # shape: (n_classes, n_features)
    n_features = coef_scaled.shape[1]

    if feature_names is None:
        feature_names = [f"latent_{i}" for i in range(n_features)]

    classes = lr.classes_

    # Scaled coefficients (comparable across features since StandardScaler normalizes)
    df_scaled = pd.DataFrame(coef_scaled, index=classes, columns=feature_names)

    # Original-space coefficients (if scaler available)
    if scaler is not None and hasattr(scaler, "scale_"):
        coef_original = coef_scaled / scaler.scale_[np.newaxis, :]
        df_original = pd.DataFrame(coef_original, index=classes, columns=feature_names)
    else:
        df_original = df_scaled.copy()

    return df_scaled, df_original


def compute_wald_pvalues(
    lr: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    """Compute Wald-test p-values for each coefficient.

    Uses the diagonal of the inverse Hessian (Fisher information) to
    estimate standard errors, then z = coef / SE, p = 2 * Phi(-|z|).

    Parameters
    ----------
    lr : fitted LogisticRegression
    X : training features (original space, before scaling)
    y : training labels for this target
    scaler : if provided, X is scaled before computing the Hessian

    Returns
    -------
    DataFrame with columns: class, feature, coefficient, std_error, z_stat, p_value
    """
    if scaler is not None:
        X_used = scaler.transform(X)
    else:
        X_used = X

    classes = lr.classes_
    n_classes = len(classes)
    n_features = X_used.shape[1]

    # For multinomial LR, compute per-class probabilities
    probs = lr.predict_proba(X_used)  # (n_samples, n_classes)

    rows = []
    for k in range(n_classes):
        # Weight matrix for class k: W_k = diag(p_k * (1 - p_k))
        p_k = probs[:, k]
        w_k = p_k * (1 - p_k)

        # Fisher information: X^T W X
        # For numerical stability, use the weighted version
        XtWX = (X_used.T * w_k[np.newaxis, :]) @ X_used

        # Add regularization term (L2 penalty from LR)
        C = lr.C if hasattr(lr, "C") else 1.0
        reg = np.eye(n_features) / C
        XtWX += reg

        try:
            cov = np.linalg.inv(XtWX)
            se = np.sqrt(np.maximum(np.diag(cov), 0))
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            cov = np.linalg.pinv(XtWX)
            se = np.sqrt(np.maximum(np.diag(cov), 0))

        coefs = lr.coef_[k]
        z_stats = np.where(se > 0, coefs / se, 0.0)
        p_values = 2 * sp_stats.norm.sf(np.abs(z_stats))

        for j in range(n_features):
            rows.append({
                "class": classes[k],
                "feature_index": j,
                "feature": f"latent_{j}",
                "coefficient": coefs[j],
                "std_error": se[j],
                "z_stat": z_stats[j],
                "p_value": p_values[j],
            })

    return pd.DataFrame(rows)


def get_top_features(
    wald_df: pd.DataFrame,
    top_n: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Get the top contributing features ranked by absolute z-statistic.

    Parameters
    ----------
    wald_df : output of compute_wald_pvalues
    top_n : number of top features per class
    alpha : significance threshold for flagging
    """
    wald_df = wald_df.copy()
    wald_df["abs_z"] = wald_df["z_stat"].abs()
    wald_df["significant"] = wald_df["p_value"] < alpha

    top_per_class = (
        wald_df
        .sort_values(["class", "abs_z"], ascending=[True, False])
        .groupby("class")
        .head(top_n)
    )
    return top_per_class


# ======================================================================
# Visualization
# ======================================================================

def plot_coefficient_heatmap(
    coef_df: pd.DataFrame,
    title: str = "Logistic Regression Coefficients",
    top_n: int = 30,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of coefficients for the top-N most important features.

    Importance is measured by max absolute coefficient across classes.
    """
    max_abs = coef_df.abs().max(axis=0)
    top_features = max_abs.nlargest(top_n).index.tolist()
    subset = coef_df[top_features]

    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.5), max(4, len(subset) * 1.2)))
    sns.heatmap(
        subset, cmap="RdBu_r", center=0, annot=False,
        xticklabels=True, yticklabels=True, ax=ax,
        cbar_kws={"label": "Coefficient"},
    )
    ax.set_title(title)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Class")
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_top_features_bar(
    wald_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Top Latent Dimensions by Significance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart showing top features per class by |z-stat|."""
    top = get_top_features(wald_df, top_n=top_n)

    classes = top["class"].unique()
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6), sharey=False)
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        cls_data = top[top["class"] == cls].sort_values("abs_z", ascending=True).tail(top_n)
        colors = ["#2ecc71" if sig else "#95a5a6" for sig in cls_data["significant"]]
        ax.barh(cls_data["feature"], cls_data["abs_z"], color=colors)
        ax.set_xlabel("|z-statistic|")
        ax.set_title(f"{cls}")
        ax.axvline(x=sp_stats.norm.ppf(0.975), color="red", linestyle="--",
                    linewidth=0.8, label="α=0.05 threshold")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_coefficient_direction(
    coef_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Coefficient Direction — Top Latent Dimensions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Diverging bar chart showing coefficient sign and magnitude per class."""
    max_abs = coef_df.abs().max(axis=0)
    top_features = max_abs.nlargest(top_n).index.tolist()

    classes = coef_df.index.tolist()
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        vals = coef_df.loc[cls, top_features].sort_values()
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient (scaled)")
        ax.set_title(f"{cls}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# ======================================================================
# High-level runner
# ======================================================================

def run_lr_interpretation(
    multi_output_model: MultiOutputClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_index: int = 2,
    target_name: str = "position",
    output_dir: str = ".",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Run full LR interpretation analysis and save outputs.

    Parameters
    ----------
    multi_output_model : fitted MultiOutputClassifier containing a LR pipeline
    X_train : training features (original space)
    y_train : training labels array, shape (n_samples, n_targets) or (n_samples,)
    target_index : which target to interpret (2 = position)
    target_name : label for the target
    output_dir : directory to save CSVs and plots
    top_n : number of top features to highlight

    Returns
    -------
    dict with keys: coef_scaled_df, coef_original_df, wald_df, top_features_df
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Extract the LR model
    lr, scaler = extract_position_lr(multi_output_model, target_index=target_index)

    # Get target labels
    if y_train.ndim == 2:
        y_target = y_train[:, target_index]
    else:
        y_target = y_train

    # Coefficients
    coef_scaled, coef_original = get_coefficients(lr, scaler)

    coef_scaled.to_csv(os.path.join(output_dir, f"lr_{target_name}_coefficients_scaled.csv"))
    coef_original.to_csv(os.path.join(output_dir, f"lr_{target_name}_coefficients_original.csv"))

    # Wald test p-values
    wald_df = compute_wald_pvalues(lr, X_train, y_target, scaler)
    wald_df.to_csv(os.path.join(output_dir, f"lr_{target_name}_wald_test.csv"), index=False)

    # Top features
    top_df = get_top_features(wald_df, top_n=top_n)
    top_df.to_csv(os.path.join(output_dir, f"lr_{target_name}_top_features.csv"), index=False)

    # Summary: significant features per class
    sig_counts = (
        wald_df[wald_df["p_value"] < 0.05]
        .groupby("class")
        .size()
        .to_dict()
    )
    total_features = lr.coef_.shape[1]

    summary_lines = [
        f"# Logistic Regression Interpretation: {target_name.title()}",
        "",
        f"**Classes:** {', '.join(lr.classes_)}",
        f"**Total latent dimensions:** {total_features}",
        "",
        "## Significant Features (p < 0.05)",
        "",
        "| Class | Significant Dims | % of Total |",
        "|-------|-----------------|------------|",
    ]
    for cls in lr.classes_:
        n_sig = sig_counts.get(cls, 0)
        pct = 100 * n_sig / total_features
        summary_lines.append(f"| {cls} | {n_sig} | {pct:.1f}% |")

    summary_lines.append("")
    summary_lines.append(f"## Top {top_n} Most Significant Dimensions Per Class")
    summary_lines.append("")

    for cls in lr.classes_:
        cls_top = top_df[top_df["class"] == cls].head(top_n)
        summary_lines.append(f"### {cls}")
        summary_lines.append("")
        summary_lines.append("| Rank | Dimension | Coefficient | z-stat | p-value | Significant |")
        summary_lines.append("|------|-----------|-------------|--------|---------|-------------|")
        for rank, (_, row) in enumerate(cls_top.iterrows(), 1):
            sig_marker = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
            summary_lines.append(
                f"| {rank} | {row['feature']} | {row['coefficient']:.4f} | "
                f"{row['z_stat']:.3f} | {row['p_value']:.2e} | {sig_marker} |"
            )
        summary_lines.append("")

    # Interpretation guide
    summary_lines.extend([
        "## Interpretation Guide",
        "",
        "- **Coefficient > 0**: higher values in this latent dimension increase "
        "the log-odds of predicting this class",
        "- **Coefficient < 0**: higher values decrease the log-odds",
        "- **|z-stat|**: magnitude of evidence against the null (coef = 0); "
        "larger = more significant",
        "- **p-value**: probability of observing this z-stat if the true coefficient "
        "were zero",
        "- Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001",
        "- Coefficients are in *scaled* feature space (after StandardScaler), "
        "so magnitudes are directly comparable across dimensions",
        "",
    ])

    with open(os.path.join(output_dir, f"lr_{target_name}_interpretation.md"), "w") as f:
        f.write("\n".join(summary_lines))

    # Plots
    plot_coefficient_heatmap(
        coef_scaled, title=f"LR Coefficients — {target_name.title()} (scaled)",
        top_n=top_n,
        save_path=os.path.join(output_dir, f"lr_{target_name}_coef_heatmap.png"),
    )
    plot_top_features_bar(
        wald_df, top_n=top_n,
        title=f"Top Latent Dimensions — {target_name.title()} (by |z-stat|)",
        save_path=os.path.join(output_dir, f"lr_{target_name}_top_features_bar.png"),
    )
    plot_coefficient_direction(
        coef_scaled, top_n=top_n,
        title=f"Coefficient Direction — {target_name.title()} (top {top_n})",
        save_path=os.path.join(output_dir, f"lr_{target_name}_coef_direction.png"),
    )

    return {
        "coef_scaled_df": coef_scaled,
        "coef_original_df": coef_original,
        "wald_df": wald_df,
        "top_features_df": top_df,
        "sig_counts": sig_counts,
    }
