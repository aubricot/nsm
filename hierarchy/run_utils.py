"""
Utilities for creating versioned, non-overwriting run directories
and generating run manifests (README.md files) for each experiment.
"""
import os
import sys
import json
import datetime
from typing import Dict, List, Optional, Any


def create_run_directory(base_dir: str = "results", prefix: str = "run") -> str:
    """
    Creates a uniquely named, timestamped run directory that will never
    overwrite a previous run.

    Format: {base_dir}/{prefix}_YYYYMMDD_HHMMSS[_N]

    If two runs happen in the same second an incrementing suffix _2, _3, …
    is appended.

    Returns the absolute path to the newly created directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{prefix}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)

    # Ensure uniqueness
    counter = 2
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, f"{run_name}_{counter}")
        counter += 1

    os.makedirs(run_dir, exist_ok=True)
    return os.path.abspath(run_dir)


def write_run_manifest(
    run_dir: str,
    *,
    description: str,
    approach: str,
    script_path: str,
    train_data_paths: Optional[List[str]] = None,
    test_data_paths: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    classifier_names: Optional[List[str]] = None,
    metric_learning_method: Optional[str] = None,
    checkpoint: Optional[str] = None,
    notes: Optional[str] = None,
    extra_files: Optional[Dict[str, str]] = None,
) -> str:
    """
    Writes a README.md manifest inside *run_dir* documenting everything
    about the run: what was executed, what data was used, what files were
    produced, etc.

    Parameters
    ----------
    run_dir : str
        Absolute path to the run directory.
    description : str
        Human readable description of what this run represents.
    approach : str
        The classification approach (e.g. "Supervised + NCA Metric Learning").
    script_path : str
        Path to the script that was executed.
    train_data_paths : list of str, optional
        Paths to training data files.
    test_data_paths : list of str, optional
        Paths to test / validation data files.
    config : dict, optional
        Model / experiment configuration dictionary.
    classifier_names : list of str, optional
        Names of classifiers trained in this run.
    metric_learning_method : str, optional
        Metric learning method used (e.g. "NCA", "LMNN", or None).
    checkpoint : str, optional
        Model checkpoint identifier.
    notes : str, optional
        Any additional free-form notes.
    extra_files : dict of str -> str, optional
        Mapping of filename -> description for files produced in this run.

    Returns
    -------
    str
        Path to the written README.md.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Run Manifest",
        "",
        f"**Created:** {timestamp}  ",
        f"**Description:** {description}  ",
        f"**Approach:** {approach}  ",
        f"**Script:** `{os.path.basename(script_path)}`  ",
    ]

    if checkpoint:
        lines.append(f"**Model Checkpoint:** `{checkpoint}`  ")
    if metric_learning_method:
        lines.append(f"**Metric Learning:** {metric_learning_method}  ")

    lines.append("")

    # --- Classifiers ---
    if classifier_names:
        lines.append("## Classifiers Trained")
        lines.append("")
        for name in classifier_names:
            lines.append(f"- {name}")
        lines.append("")

    # --- Training Data ---
    if train_data_paths:
        lines.append("## Training Data")
        lines.append("")
        lines.append(f"Total training files: **{len(train_data_paths)}**")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand full list</summary>")
        lines.append("")
        for p in train_data_paths:
            lines.append(f"- `{os.path.basename(p)}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # --- Test Data ---
    if test_data_paths:
        lines.append("## Test / Validation Data")
        lines.append("")
        lines.append(f"Total test files: **{len(test_data_paths)}**")
        lines.append("")
        for p in test_data_paths:
            lines.append(f"- `{os.path.basename(p)}`")
        lines.append("")

    # --- Configuration ---
    if config:
        lines.append("## Configuration")
        lines.append("")
        # Write a subset of config keys that are most relevant
        safe_keys = [
            'latent_size', 'n_pts_per_object', 'samples_per_object_per_batch',
            'percent_near_surface', 'percent_further_from_surface',
            'sigma_near', 'sigma_far', 'device', 'learning_rate',
        ]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k in safe_keys:
            if k in config:
                lines.append(f"| `{k}` | `{config[k]}` |")
        lines.append("")

    # --- Files Produced ---
    if extra_files:
        lines.append("## Files in this Directory")
        lines.append("")
        lines.append("| File | Description |")
        lines.append("|------|-------------|")
        for fname, desc in extra_files.items():
            lines.append(f"| `{fname}` | {desc} |")
        lines.append("")

    # --- Notes ---
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
        lines.append("")

    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return readme_path


def update_manifest_files_table(run_dir: str, extra_files: Dict[str, str]) -> None:
    """
    Appends or updates the 'Files in this Directory' section of an existing
    README.md in *run_dir*.  If the section doesn't exist it creates it.
    """
    readme_path = os.path.join(run_dir, "README.md")
    if not os.path.exists(readme_path):
        # Create a minimal manifest
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Run Manifest\n\n")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build the new table
    table_lines = [
        "",
        "## Files in this Directory",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for fname, desc in extra_files.items():
        table_lines.append(f"| `{fname}` | {desc} |")
    table_lines.append("")
    new_section = "\n".join(table_lines)

    # Replace existing section if present
    import re
    pattern = r"## Files in this Directory.*?(?=\n## |\Z)"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_section.strip(), content, flags=re.DOTALL)
    else:
        content += "\n" + new_section

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
