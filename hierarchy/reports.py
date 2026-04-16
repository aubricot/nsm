"""
Generate comprehensive classification reports with all metrics specified
in Project Proposal V2.

This script reads a summary CSV produced by classify_vertebrae_v3.py,
computes all required metrics for each classification approach, generates
comparison visuals and confusion matrices, and writes everything into a
timestamped, non-overwriting results directory complete with a README.md
manifest.

Usage:
    python generate_reports.py <path_to_summary_csv>

If no CSV is provided it falls back to a mock dataset for demonstration.
"""
import os
import sys
import argparse
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

from .evaluation import (
    calculate_metrics,
    metrics_to_dataframe,
    generate_hierarchical_confusion_matrices,
    generate_position_confusion_matrix,
)
from .taxonomy import parse_taxonomy_from_filename
from .run_utils import create_run_directory, write_run_manifest, update_manifest_files_table

warnings.filterwarnings("ignore")

CLASSIFIERS = ["KNN", "SVM", "RandomForest", "MLP", "LogisticRegression"]
DISTANCE_METHODS = ["cos", "euc"]


# ======================================================================
# Data loading
# ======================================================================

def load_summary_csv(csv_path: str) -> pd.DataFrame:
    """Load a real summary CSV produced by classify_vertebrae_v3."""
    df = pd.read_csv(csv_path)
    return df


def generate_mock_data() -> pd.DataFrame:
    """Creates a mock dataset for demonstration / testing when no real CSV
    is available."""
    print("Generating mock dataset for demonstration...")
    np.random.seed(42)
    n = 120

    families  = (["Scincidae"] * 40 + ["Cordylidae"] * 40 + ["Agamidae"] * 40)
    genera    = (["Tribolonotus"] * 20 + ["Corucia"] * 20 +
                 ["Smaug"] * 20 + ["Cordylus"] * 20 +
                 ["Pogona"] * 20 + ["Amphibolurus"] * 20)
    species   = (["novaeguineae"] * 20 + ["zebrata"] * 20 +
                 ["giganteus"] * 20 + ["jonesii"] * 20 +
                 ["vitticeps"] * 20 + ["muricatus"] * 20)
    positions = np.random.choice(["Cervical", "Thoracic", "Lumbar"], n)

    meshes = [
        f"{fam}_{gen}_{sp}_spec{i}-c1.vtk"
        for i, (fam, gen, sp) in enumerate(zip(families, genera, species))
    ]

    df = pd.DataFrame({
        "mesh": meshes,
        "ground_truth_species": species,
        "ground_truth_position": positions,
    })

    # Add mock predictions for each classifier (species + position)
    unique_sp = list(set(species))
    pos_choices = ["Cervical", "Thoracic", "Lumbar"]
    error_rates = {"KNN": 0.10, "SVM": 0.05, "RandomForest": 0.15,
                   "MLP": 0.08, "LogisticRegression": 0.20}

    for clf in CLASSIFIERS:
        err = error_rates[clf]
        # Species predictions
        sp_preds = np.where(
            np.random.rand(n) > err,
            species,
            np.random.choice(unique_sp, n),
        )
        df[f"{clf}_predicted_species"] = sp_preds
        df[f"{clf}_species_confidence"] = [
            f"{np.random.uniform(0.6, 1.0):.2%}" for _ in range(n)
        ]

        # Position predictions
        pos_preds = np.where(
            np.random.rand(n) > err,
            positions,
            np.random.choice(pos_choices, n),
        )
        df[f"{clf}_predicted_position"] = pos_preds
        df[f"{clf}_position_confidence"] = [
            f"{np.random.uniform(0.5, 1.0):.2%}" for _ in range(n)
        ]

        df[f"{clf}_inf_time"] = np.random.normal(0.01, 0.003, n).clip(0.001)
        df[f"{clf}_match_species"] = np.where(
            df["ground_truth_species"] == df[f"{clf}_predicted_species"], "yes", "no"
        )
        df[f"{clf}_match_position"] = np.where(
            df["ground_truth_position"] == df[f"{clf}_predicted_position"], "yes", "no"
        )

    # Distance-based top-1 / top-5
    for method in DISTANCE_METHODS:
        df[f"{method}_top1_species"] = np.where(
            np.random.rand(n) > 0.12, species, np.random.choice(unique_sp, n)
        )
        df[f"{method}_top1_match"] = np.where(
            df["ground_truth_species"] == df[f"{method}_top1_species"], "yes", "no"
        )
        df[f"{method}_top5_match"] = np.where(
            np.random.rand(n) > 0.05, "yes", "no"
        )

    return df


# ======================================================================
# Metric computation
# ======================================================================

def compute_all_classifier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every classifier in CLASSIFIERS, compute all metrics specified in
    the Project Proposal V2 for both species and position prediction.

    Returns a tidy DataFrame.
    """
    all_rows = []

    for clf in CLASSIFIERS:
        sp_col = f"{clf}_predicted_species"
        pos_col = f"{clf}_predicted_position"

        # --- Species metrics ---
        if sp_col in df.columns:
            y_true = df["ground_truth_species"].tolist()
            y_pred = df[sp_col].tolist()
            sp_metrics = calculate_metrics(y_true, y_pred)
            sp_df = metrics_to_dataframe(sp_metrics, classifier_name=clf)
            sp_df["target"] = "species"
            all_rows.append(sp_df)

        # --- Position metrics ---
        if pos_col in df.columns and "ground_truth_position" in df.columns:
            mask = df["ground_truth_position"].notna() & df[pos_col].notna()
            y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
            y_pred_pos = df.loc[mask, pos_col].tolist()
            if len(y_true_pos) > 0:
                pos_metrics = calculate_metrics(y_true_pos, y_pred_pos)
                pos_df = metrics_to_dataframe(pos_metrics, classifier_name=clf)
                pos_df["target"] = "position"
                all_rows.append(pos_df)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def compute_distance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes simple accuracy metrics for the cosine / euclidean distance
    based approaches (top-1 and top-5).
    """
    rows = []
    for method in DISTANCE_METHODS:
        label = "Cosine Similarity" if method == "cos" else "Euclidean Distance"

        top1_col = f"{method}_top1_match"
        top5_col = f"{method}_top5_match"

        if top1_col in df.columns:
            top1_acc = (df[top1_col] == "yes").mean()
            rows.append({"method": label, "metric": "top_1_accuracy", "value": top1_acc})

        if top5_col in df.columns:
            top5_acc = (df[top5_col] == "yes").mean()
            rows.append({"method": label, "metric": "top_5_accuracy", "value": top5_acc})

    return pd.DataFrame(rows)


# ======================================================================
# Visualizations
# ======================================================================

def plot_performance_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart: Accuracy / Avg Class Acc / F1 (macro) per classifier."""
    os.makedirs(output_dir, exist_ok=True)

    y_true = df["ground_truth_species"].tolist()
    records = []
    for clf in CLASSIFIERS:
        col = f"{clf}_predicted_species"
        if col not in df.columns:
            continue
        y_pred = df[col].tolist()
        m = calculate_metrics(y_true, y_pred)
        records.append({"Classifier": clf, "Metric": "Instance Accuracy",
                        "Score": m["instance_accuracy"]})
        records.append({"Classifier": clf, "Metric": "Avg Class Accuracy",
                        "Score": m["average_class_accuracy"]})
        records.append({"Classifier": clf, "Metric": "Precision (macro)",
                        "Score": m["precision_macro"]})
        records.append({"Classifier": clf, "Metric": "Recall (macro)",
                        "Score": m["recall_macro"]})
        records.append({"Classifier": clf, "Metric": "F1 (macro)",
                        "Score": m["f1_macro"]})

    if not records:
        return

    plot_df = pd.DataFrame(records)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x="Classifier", y="Score", hue="Metric",
                palette="viridis")
    plt.title("Classifier Performance Comparison — Species Level")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison_species.png"), dpi=300)
    plt.close()

    # Position-level chart (if available)
    if "ground_truth_position" in df.columns:
        pos_records = []
        for clf in CLASSIFIERS:
            col = f"{clf}_predicted_position"
            if col not in df.columns:
                continue
            mask = df["ground_truth_position"].notna() & df[col].notna()
            y_t = df.loc[mask, "ground_truth_position"].tolist()
            y_p = df.loc[mask, col].tolist()
            if len(y_t) == 0:
                continue
            m = calculate_metrics(y_t, y_p)
            pos_records.append({"Classifier": clf, "Metric": "Instance Accuracy",
                                "Score": m["instance_accuracy"]})
            pos_records.append({"Classifier": clf, "Metric": "Avg Class Accuracy",
                                "Score": m["average_class_accuracy"]})
            pos_records.append({"Classifier": clf, "Metric": "F1 (macro)",
                                "Score": m["f1_macro"]})

        if pos_records:
            pos_plot_df = pd.DataFrame(pos_records)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=pos_plot_df, x="Classifier", y="Score", hue="Metric",
                        palette="Set2")
            plt.title("Classifier Performance Comparison — Spinal Position")
            plt.ylim(0, 1.05)
            plt.ylabel("Score")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "performance_comparison_position.png"), dpi=300)
            plt.close()


def plot_inference_times(df: pd.DataFrame, output_dir: str):
    """Box plot showing inference time distributions per classifier."""
    os.makedirs(output_dir, exist_ok=True)
    time_data = []
    for clf in CLASSIFIERS:
        col = f"{clf}_inf_time"
        if col in df.columns:
            for t in df[col]:
                time_data.append({"Classifier": clf, "Inference Time (s)": float(t)})

    if not time_data:
        return

    time_df = pd.DataFrame(time_data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_df, x="Classifier", y="Inference Time (s)", palette="Set2")
    plt.title("Inference Time Distribution per Classifier")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_boxplot.png"), dpi=300)
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame, output_dir: str, classifier: str = None):
    """
    Generates hierarchical confusion matrices (family / genus / species) +
    a spinal position confusion matrix.

    Uses the best-performing classifier by default, or the one specified.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true_species = df["ground_truth_species"].tolist()

    # Pick best classifier if not specified
    if classifier is None:
        best_clf, best_acc = "KNN", 0
        for clf in CLASSIFIERS:
            col = f"{clf}_predicted_species"
            if col in df.columns:
                acc = accuracy_score(y_true_species, df[col].tolist())
                if acc > best_acc:
                    best_acc, best_clf = acc, clf
        classifier = best_clf

    print(f"Generating confusion matrices for: {classifier}")

    # --- Hierarchical CMs (species / genus / family) ---
    true_dicts, pred_dicts = [], []
    species_taxonomy_map = {}

    for filename in df["mesh"]:
        parsed = parse_taxonomy_from_filename(filename)
        if parsed:
            true_dicts.append(parsed)
            species_taxonomy_map[parsed["species"]] = parsed
        else:
            true_dicts.append({"family": "unknown", "genus": "unknown", "species": "unknown"})

    col = f"{classifier}_predicted_species"
    for pred_sp in df[col]:
        if pred_sp in species_taxonomy_map:
            pred_dicts.append(species_taxonomy_map[pred_sp])
        else:
            pred_dicts.append({"family": "unknown", "genus": "unknown", "species": pred_sp})

    cm_results = generate_hierarchical_confusion_matrices(true_dicts, pred_dicts)

    for level, data in cm_results.items():
        matrix, labels = data["matrix"], data["labels"]
        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.7)))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix — {level.capitalize()} Level ({classifier})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{level}.png"), dpi=300)
        plt.close()

    # --- Spinal Position CM ---
    pos_col = f"{classifier}_predicted_position"
    if pos_col in df.columns and "ground_truth_position" in df.columns:
        mask = df["ground_truth_position"].notna() & df[pos_col].notna()
        y_true_pos = df.loc[mask, "ground_truth_position"].tolist()
        y_pred_pos = df.loc[mask, pos_col].tolist()
        if y_true_pos:
            pos_cm = generate_position_confusion_matrix(y_true_pos, y_pred_pos)
            labels = pos_cm["labels"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(pos_cm["matrix"], annot=True, fmt="d", cmap="Oranges",
                        xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix — Spinal Position ({classifier})")
            plt.xlabel("Predicted Position")
            plt.ylabel("True Position")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix_position.png"), dpi=300)
            plt.close()

    print(f"Confusion matrices saved to {output_dir}")


def plot_distance_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart comparing top-1 and top-5 accuracy for distance methods."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for method in DISTANCE_METHODS:
        label = "Cosine Similarity" if method == "cos" else "Euclidean Distance"
        t1 = f"{method}_top1_match"
        t5 = f"{method}_top5_match"
        if t1 in df.columns:
            records.append({"Method": label, "Metric": "Top-1 Accuracy",
                            "Score": (df[t1] == "yes").mean()})
        if t5 in df.columns:
            records.append({"Method": label, "Metric": "Top-5 Accuracy",
                            "Score": (df[t5] == "yes").mean()})

    if not records:
        return

    plot_df = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="Method", y="Score", hue="Metric", palette="Set1")
    plt.title("Distance-Based Retrieval Accuracy")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_retrieval_accuracy.png"), dpi=300)
    plt.close()


# ======================================================================
# Summary text report
# ======================================================================

def write_summary_text(
    metrics_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    training_times: dict,
    run_dir: str,
):
    """Writes a human-readable summary_statistics.md file."""
    path = os.path.join(run_dir, "summary_statistics.md")

    lines = [
        "# Summary Statistics",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
    ]

    # --- Supervised classifier summary (species) ---
    lines.append("## Supervised Classifiers — Species Prediction")
    lines.append("")
    sp_summary = metrics_df[
        (metrics_df["target"] == "species") & (metrics_df["level"] == "summary")
    ].copy()

    if not sp_summary.empty:
        display_cols = [
            "classifier", "instance_accuracy", "average_class_accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "n_samples",
        ]
        existing = [c for c in display_cols if c in sp_summary.columns]
        lines.append(sp_summary[existing].to_markdown(index=False))
        lines.append("")

        # Top-5 accuracy if present
        if "top_5_accuracy" in sp_summary.columns:
            lines.append("### Top-5 Accuracy")
            lines.append("")
            for _, row in sp_summary.iterrows():
                val = row.get("top_5_accuracy", "N/A")
                lines.append(f"- **{row['classifier']}**: {val}")
            lines.append("")

    # --- Supervised classifier summary (position) ---
    pos_summary = metrics_df[
        (metrics_df["target"] == "position") & (metrics_df["level"] == "summary")
    ].copy()

    if not pos_summary.empty:
        lines.append("## Supervised Classifiers — Spinal Position Prediction")
        lines.append("")
        display_cols = [
            "classifier", "instance_accuracy", "average_class_accuracy",
            "precision_macro", "recall_macro", "f1_macro", "n_samples",
        ]
        existing = [c for c in display_cols if c in pos_summary.columns]
        lines.append(pos_summary[existing].to_markdown(index=False))
        lines.append("")

    # --- Distance methods ---
    if not distance_df.empty:
        lines.append("## Distance-Based Retrieval")
        lines.append("")
        lines.append(distance_df.to_markdown(index=False))
        lines.append("")

    # --- Training times ---
    if training_times:
        lines.append("## Training Times")
        lines.append("")
        lines.append("| Classifier | Training Time (s) |")
        lines.append("|------------|-------------------|")
        for clf, t in training_times.items():
            lines.append(f"| {clf} | {t:.4f} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate classification reports.")
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to summary CSV from classify_vertebrae_v3.py")
    parser.add_argument("--training-times", default=None,
                        help="Path to training_times.json (optional)")
    parser.add_argument("--base-dir", default="results",
                        help="Base directory for output runs (default: results)")
    args = parser.parse_args()

    # Load data
    if args.csv and os.path.isfile(args.csv):
        print(f"Loading summary CSV: {args.csv}")
        df = load_summary_csv(args.csv)
    else:
        df = generate_mock_data()

    # Create run directory (never overwrites)
    run_dir = create_run_directory(base_dir=args.base_dir, prefix="report")
    print(f"Results will be saved to: {run_dir}")

    # Sub-directories
    charts_dir = os.path.join(run_dir, "charts")
    cm_dir = os.path.join(run_dir, "confusion_matrices")

    # Compute metrics
    print("Computing classification metrics...")
    metrics_df = compute_all_classifier_metrics(df)
    distance_df = compute_distance_metrics(df)

    # Load training times if available
    training_times = {}
    if args.training_times and os.path.isfile(args.training_times):
        with open(args.training_times) as f:
            training_times = json.load(f)

    # Save detailed metrics CSV
    metrics_csv = os.path.join(run_dir, "detailed_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Detailed metrics saved to: {metrics_csv}")

    distance_csv = os.path.join(run_dir, "distance_metrics.csv")
    distance_df.to_csv(distance_csv, index=False)

    # Generate visuals
    print("Generating visualizations...")
    plot_performance_comparison(df, charts_dir)
    plot_inference_times(df, charts_dir)
    plot_confusion_matrices(df, cm_dir)
    plot_distance_comparison(df, charts_dir)

    # Summary text
    summary_path = write_summary_text(metrics_df, distance_df, training_times, run_dir)
    print(f"Summary statistics written to: {summary_path}")

    # Write manifest
    produced_files = {
        "README.md": "Run manifest describing this report directory",
        "summary_statistics.md": "Human-readable summary of all metrics",
        "detailed_metrics.csv": "Full per-classifier, per-class metrics",
        "distance_metrics.csv": "Top-1/Top-5 accuracy for distance methods",
        "charts/performance_comparison_species.png": "Bar chart — species metrics per classifier",
        "charts/inference_time_boxplot.png": "Box plot — inference times",
        "charts/distance_retrieval_accuracy.png": "Bar chart — distance retrieval accuracy",
        "confusion_matrices/confusion_matrix_species.png": "Species-level confusion matrix",
        "confusion_matrices/confusion_matrix_genus.png": "Genus-level confusion matrix",
        "confusion_matrices/confusion_matrix_family.png": "Family-level confusion matrix",
    }
    if "ground_truth_position" in df.columns:
        produced_files["charts/performance_comparison_position.png"] = (
            "Bar chart — position metrics per classifier"
        )
        produced_files["confusion_matrices/confusion_matrix_position.png"] = (
            "Spinal position confusion matrix"
        )

    write_run_manifest(
        run_dir,
        description="Classification report with all metrics from Project Proposal V2",
        approach="Supervised Classifiers + Distance-Based Retrieval",
        script_path=os.path.abspath(__file__),
        classifier_names=CLASSIFIERS,
        notes="Generated by generate_reports.py",
        extra_files=produced_files,
    )

    print(f"\nDone! All results saved to: {run_dir}")


if __name__ == "__main__":
    main()
