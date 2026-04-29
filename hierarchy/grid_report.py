"""Generate a comparison report across grid-search run directories.

Reads every `hierarchy_grid_<sweep_ts>_cw<X>_hw<Y>/` directory under
`hierarchy/grid_search_results/` (or another path), aggregates each run's
classification metrics + hyperparams, and writes a comparison report
(markdown + CSV + JSON + heatmap PNGs) to a fresh timestamped subdir.

Run with::

    python run_grid_report.py [--results-dir DIR] [--output-dir DIR]
                              [--top-n N] [--include-baselines]
                              [--metrics METRIC1,METRIC2,...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .run_utils import create_run_directory, write_run_manifest


# ---- Constants -------------------------------------------------------------

RUN_NAME_RE = re.compile(
    r"^hierarchy_grid_(?P<sweep_ts>\d{8}_\d{6})_cw(?P<cw>[\d.]+)_hw(?P<hw>[\d.]+)"
    r"(?:_(?P<margin_tag>m[a-z0-9]+))?$"
)

# (metric_column, ablation_config, human_label)
DEFAULT_HEADLINE_METRICS: List[Tuple[str, str, str]] = [
    ("cosine_species_top_1_accuracy", "A_baseline",
     "Cosine top-1 species (raw latent)"),
    ("cosine_genus_top_1_accuracy", "A_baseline",
     "Cosine top-1 genus (raw latent)"),
    ("KNN_species_accuracy", "D_full_pipeline",
     "KNN species accuracy (full pipeline)"),
    ("KNN_species_f1_macro", "D_full_pipeline",
     "KNN species F1-macro (full pipeline)"),
    ("KNN_position_accuracy", "D_full_pipeline",
     "KNN position accuracy (full pipeline)"),
    ("SVM_species_accuracy", "D_full_pipeline",
     "SVM species accuracy (full pipeline)"),
]

ID_COLS = [
    'run_name', 'sweep_ts', 'is_baseline', 'ablation_config', 'ablation_label',
    'margin_tag', 'contrastive_margins',
    'contrastive_weight', 'head_weight',
    'species_weight', 'genus_weight', 'family_weight', 'position_weight',
    'contrastive_warmup', 'head_warmup', 'classification_head_hidden_dim',
    'epochs_trained',
    'final_l1_loss', 'final_hierarchy_contrastive_loss',
    'final_classification_head_loss',
]


# ---- Per-run ingestion -----------------------------------------------------

def _parse_run_name(name: str) -> Dict[str, Optional[Any]]:
    """Extract sweep_ts, cw, hw, and optional margin tag from a run dir name."""
    m = RUN_NAME_RE.match(name)
    if not m:
        return {'sweep_ts': None, 'contrastive_weight': None,
                'head_weight': None, 'margin_tag': None}
    return {
        'sweep_ts': m.group('sweep_ts'),
        'contrastive_weight': float(m.group('cw')),
        'head_weight': float(m.group('hw')),
        'margin_tag': m.group('margin_tag'),  # e.g. "m0248", or None for default sweep
    }


def _load_hyperparams(run_dir: str) -> Dict[str, Any]:
    """Load hyperparams.json if present; return empty dict otherwise."""
    path = os.path.join(run_dir, 'hyperparams.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _find_latest_ablation(run_dir: str) -> Optional[str]:
    """Return the lexicographically latest results/ablation_*/ path, or None."""
    candidates = sorted(glob.glob(os.path.join(run_dir, 'results', 'ablation_*')))
    candidates = [c for c in candidates if os.path.isdir(c)]
    return candidates[-1] if candidates else None


def _load_train_log_summary(run_dir: str) -> Dict[str, Any]:
    """Read the last row of train_log.csv for final-loss values."""
    path = os.path.join(run_dir, 'train_log.csv')
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        'epochs_trained': int(df['epoch'].iloc[-1]) if 'epoch' in df.columns else len(df),
        'final_l1_loss': float(last.get('l1_loss', np.nan)),
        'final_hierarchy_contrastive_loss': float(
            last.get('hierarchy_contrastive_loss', np.nan)),
        'final_classification_head_loss': float(
            last.get('classification_head_loss', np.nan)),
    }


def _ingest_run(run_dir: str, *, is_baseline: bool = False
                ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single run dir.

    Returns (rows, incomplete_record).
        rows: list of dicts, one per ablation row (4 if COMPLETE, 0 otherwise).
        incomplete_record: dict with run_name + reason if the run was excluded;
            None if the run was COMPLETE.
    """
    run_name = os.path.basename(run_dir.rstrip('/'))
    hp = _load_hyperparams(run_dir)
    parsed = _parse_run_name(run_name)

    # Combine: hyperparams.json wins; fall back to regex.
    cw = hp.get('hierarchy_contrastive_weight', parsed.get('contrastive_weight'))
    hw = hp.get('classification_head_weight', parsed.get('head_weight'))
    sweep_ts = hp.get('timestamp', parsed.get('sweep_ts'))

    level_weights = hp.get('classification_level_weights') or {}

    margins = hp.get('hierarchy_contrastive_margins')
    base_id = {
        'run_name': run_name,
        'sweep_ts': sweep_ts,
        'is_baseline': is_baseline,
        'margin_tag': parsed.get('margin_tag'),
        'contrastive_margins': (json.dumps(margins, sort_keys=True)
                                if margins is not None else None),
        'contrastive_weight': cw,
        'head_weight': hw,
        'species_weight': level_weights.get('species'),
        'genus_weight': level_weights.get('genus'),
        'family_weight': level_weights.get('family'),
        'position_weight': level_weights.get('position'),
        'contrastive_warmup': hp.get('hierarchy_contrastive_warmup'),
        'head_warmup': hp.get('classification_head_warmup'),
        'classification_head_hidden_dim': hp.get('classification_head_hidden_dim'),
    }
    base_id.update(_load_train_log_summary(run_dir))

    ablation_dir = _find_latest_ablation(run_dir)
    if ablation_dir is None:
        return [], {**base_id, 'reason': 'no results/ablation_* dir'}

    cmp_path = os.path.join(ablation_dir, 'comparison_metrics.csv')
    if not os.path.isfile(cmp_path):
        return [], {**base_id, 'reason': f'no comparison_metrics.csv in {os.path.basename(ablation_dir)}'}

    try:
        cmp_df = pd.read_csv(cmp_path)
    except (pd.errors.EmptyDataError, OSError) as e:
        return [], {**base_id, 'reason': f'failed to read comparison_metrics.csv: {e}'}

    if cmp_df.empty:
        return [], {**base_id, 'reason': 'comparison_metrics.csv is empty'}

    rows = []
    for _, ablation_row in cmp_df.iterrows():
        row = dict(base_id)
        row['ablation_config'] = ablation_row.get('config', '?')
        row['ablation_label'] = ablation_row.get('label', '')
        for col, val in ablation_row.items():
            if col in ('config', 'label'):
                continue
            row[col] = val
        rows.append(row)
    return rows, None


# ---- Aggregation -----------------------------------------------------------

def _discover_run_dirs(results_dir: str) -> List[str]:
    pat = os.path.join(results_dir, 'hierarchy_grid_*')
    return sorted(d for d in glob.glob(pat) if os.path.isdir(d))


def _discover_baseline_dirs(project_root: str) -> List[str]:
    """Find hierarchy_v<N> dirs at the project root for --include-baselines."""
    out = []
    for entry in sorted(os.listdir(project_root)):
        full = os.path.join(project_root, entry)
        if not os.path.isdir(full):
            continue
        if re.match(r'^hierarchy_v\d+$', entry):
            out.append(full)
    return out


def _build_master_frame(run_dirs: List[str], baseline_dirs: List[str]
                        ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []

    for d in run_dirs:
        rows, inc = _ingest_run(d, is_baseline=False)
        all_rows.extend(rows)
        if inc is not None:
            incomplete.append(inc)

    for d in baseline_dirs:
        rows, inc = _ingest_run(d, is_baseline=True)
        all_rows.extend(rows)
        if inc is not None:
            incomplete.append(inc)

    if not all_rows:
        return pd.DataFrame(), incomplete

    df = pd.DataFrame(all_rows)
    # Order columns: identifiers first, then everything else alphabetical.
    leading = [c for c in ID_COLS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in leading)
    return df[leading + rest], incomplete


# ---- Visualization ---------------------------------------------------------

def _generate_heatmap(df: pd.DataFrame, metric: str, ablation_config: str,
                      label: str, output_path: str) -> Optional[str]:
    """
    Pivot grid runs to (cw rows × hw cols) and save a heatmap PNG.

    Returns the filename written, or None if there were too few points.
    """
    sub = df[df['ablation_config'] == ablation_config].copy()
    sub = sub.dropna(subset=['contrastive_weight', 'head_weight', metric])
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index='contrastive_weight',
        columns='head_weight',
        values=metric,
        aggfunc='mean',
    )
    if pivot.size < 4:
        return None  # too few cells for a meaningful heatmap

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.4),
                                    max(4, pivot.shape[0] * 1.0)))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="viridis",
        cbar_kws={'label': metric}, ax=ax,
    )
    ax.set_title(f"{label}\n[{ablation_config}] {metric}")
    ax.set_xlabel("head_weight")
    ax.set_ylabel("contrastive_weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


# ---- Markdown report -------------------------------------------------------

def _md_table(df: pd.DataFrame, *, max_rows: Optional[int] = None,
              float_fmt: str = ".4f") -> str:
    """Render a DataFrame as a markdown table, gracefully empty-handled."""
    if df.empty:
        return "_(no rows)_"
    show = df if max_rows is None else df.head(max_rows)
    try:
        return show.to_markdown(index=False, floatfmt=float_fmt)
    except ImportError:
        # pandas needs `tabulate` for to_markdown; fall back to a simple csv.
        return "```\n" + show.to_csv(index=False) + "\n```"


def _leaderboard_section(df: pd.DataFrame,
                         metrics: List[Tuple[str, str, str]],
                         top_n: int) -> str:
    out: List[str] = []
    if df.empty or 'ablation_config' not in df.columns:
        return "_(no completed runs to rank)_"
    for metric, ablation, label in metrics:
        out.append(f"\n### {label}")
        out.append(f"_{metric}_ on `{ablation}`")
        out.append("")
        sub = df[df['ablation_config'] == ablation].copy()
        if metric not in sub.columns:
            out.append(f"_(metric `{metric}` not present in any run's CSV)_")
            continue
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            out.append("_(no runs report this metric)_")
            continue
        sub = sub.sort_values(metric, ascending=False)
        cols = [c for c in
                ['run_name', 'contrastive_weight', 'head_weight', metric]
                if c in sub.columns]
        out.append(_md_table(sub[cols], max_rows=top_n))
    return "\n".join(out)


def _sensitivity_section(df: pd.DataFrame,
                         metrics: List[Tuple[str, str, str]]) -> str:
    """Marginal mean ± std across cw rows and hw columns."""
    out: List[str] = []
    if df.empty or 'ablation_config' not in df.columns:
        return "_(no completed runs for sensitivity analysis)_"
    grid_only = df[df.get('is_baseline', False) == False]  # noqa: E712
    if grid_only.empty:
        return "_(no grid runs with classification results yet — sensitivity needs ≥2 points)_"
    for metric, ablation, label in metrics:
        sub = grid_only[grid_only['ablation_config'] == ablation]
        if metric not in sub.columns:
            continue
        sub = sub.dropna(subset=['contrastive_weight', 'head_weight', metric])
        if len(sub) < 2:
            continue

        cw_marginal = (sub.groupby('contrastive_weight')[metric]
                       .agg(['mean', 'std', 'count'])
                       .reset_index()
                       .rename(columns={'contrastive_weight': 'cw'}))
        hw_marginal = (sub.groupby('head_weight')[metric]
                       .agg(['mean', 'std', 'count'])
                       .reset_index()
                       .rename(columns={'head_weight': 'hw'}))

        out.append(f"\n### {label}")
        out.append(f"_{metric}_ on `{ablation}`")
        out.append("")
        out.append("Marginal across `contrastive_weight`:")
        out.append("")
        out.append(_md_table(cw_marginal))
        out.append("")
        out.append("Marginal across `head_weight`:")
        out.append("")
        out.append(_md_table(hw_marginal))
    return "\n".join(out)


def _coverage_section(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no completed runs)_"
    cw_vals = sorted(df['contrastive_weight'].dropna().unique().tolist())
    hw_vals = sorted(df['head_weight'].dropna().unique().tolist())
    sweep_ts_vals = sorted(df['sweep_ts'].dropna().unique().tolist())
    n_runs = df['run_name'].nunique()
    return (
        f"- Completed runs (rows ÷ 4 ablations ≈): **{n_runs}**\n"
        f"- Distinct `contrastive_weight` values: {cw_vals}\n"
        f"- Distinct `head_weight` values: {hw_vals}\n"
        f"- Distinct sweep timestamps: {sweep_ts_vals}\n"
    )


def _heatmaps_section(heatmap_paths: List[Tuple[str, str]]) -> str:
    if not heatmap_paths:
        return "_(no heatmaps produced — grid coverage too sparse)_"
    out: List[str] = []
    for label, rel_path in heatmap_paths:
        out.append(f"### {label}\n")
        out.append(f"![{label}]({rel_path})\n")
    return "\n".join(out)


def _incomplete_section(incomplete: List[Dict[str, Any]]) -> str:
    if not incomplete:
        return "_(none — every run produced a comparison_metrics.csv)_"
    df = pd.DataFrame([
        {'run_name': i.get('run_name'),
         'cw': i.get('contrastive_weight'),
         'hw': i.get('head_weight'),
         'epochs_trained': i.get('epochs_trained'),
         'reason': i.get('reason')}
        for i in incomplete
    ])
    return _md_table(df)


def _write_report(df: pd.DataFrame, incomplete: List[Dict[str, Any]],
                  metrics: List[Tuple[str, str, str]], top_n: int,
                  heatmap_paths: List[Tuple[str, str]],
                  output_dir: str, results_dir: str) -> str:
    md = []
    md.append(f"# Grid Search Comparison Report\n")
    md.append(f"_Generated from `{results_dir}`_\n")
    md.append("\n---\n\n## 1. Sweep Overview\n")
    md.append(_coverage_section(df))

    md.append("\n## 2. Headline Leaderboards (top {})\n".format(top_n))
    md.append(_leaderboard_section(df, metrics, top_n))

    md.append("\n## 3. Heatmaps\n")
    md.append(_heatmaps_section(heatmap_paths))

    md.append("\n## 4. Sensitivity Analysis\n")
    md.append(_sensitivity_section(df, metrics))

    md.append("\n## 5. Incomplete Runs\n")
    md.append(_incomplete_section(incomplete))

    md.append("\n## 6. Full Master Table (compact)\n")
    if df.empty:
        md.append("_(empty — see incomplete-runs section above)_\n")
    else:
        compact_cols = [c for c in
                        ['run_name', 'ablation_config', 'contrastive_weight',
                         'head_weight', 'cosine_species_top_1_accuracy',
                         'KNN_species_accuracy', 'KNN_species_f1_macro',
                         'KNN_position_accuracy', 'SVM_species_accuracy']
                        if c in df.columns]
        sort_cols = [c for c in ['run_name', 'ablation_config'] if c in compact_cols]
        view = df[compact_cols].sort_values(sort_cols) if sort_cols else df[compact_cols]
        md.append(_md_table(view))
        md.append("\n_Full per-row metrics in `master_grid_comparison.csv`._\n")

    out_path = os.path.join(output_dir, 'grid_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    return out_path


# ---- Argparse & main -------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate grid-search runs into a comparison report."
    )
    parser.add_argument(
        '--results-dir', default='hierarchy/grid_search_results',
        help='Directory containing hierarchy_grid_*/ run dirs '
             '(default: hierarchy/grid_search_results).')
    parser.add_argument(
        '--output-dir', default=None,
        help='Where to write the report. Default: a fresh report_<timestamp>/ '
             'inside --results-dir.')
    parser.add_argument(
        '--top-n', type=int, default=5,
        help='Leaderboard length per metric (default: 5).')
    parser.add_argument(
        '--include-baselines', action='store_true',
        help='Also ingest hierarchy_v<N>/ dirs at the project root and tag them as baselines.')
    parser.add_argument(
        '--metrics', default=None,
        help='Comma-separated metric column names to use as headlines '
             '(default: a curated set covering distance + classifier + '
             'hierarchy levels).')
    return parser


def _resolve_metrics(metrics_arg: Optional[str]
                     ) -> List[Tuple[str, str, str]]:
    if not metrics_arg:
        return DEFAULT_HEADLINE_METRICS
    out = []
    for token in metrics_arg.split(','):
        token = token.strip()
        if not token:
            continue
        # Allow either bare metric (assume D_full_pipeline) or
        # "<metric>@<ablation>" syntax.
        if '@' in token:
            metric, ablation = token.split('@', 1)
        else:
            metric, ablation = token, 'D_full_pipeline'
        out.append((metric, ablation, f"{metric} on {ablation}"))
    return out


def main():
    args = _build_arg_parser().parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"ERROR: results dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = _resolve_metrics(args.metrics)

    # Discover & ingest
    run_dirs = _discover_run_dirs(results_dir)
    baseline_dirs = (
        _discover_baseline_dirs(project_root) if args.include_baselines else []
    )
    print(f"Found {len(run_dirs)} grid run dir(s) under {results_dir}")
    if baseline_dirs:
        print(f"Found {len(baseline_dirs)} baseline run dir(s) at project root")

    df, incomplete = _build_master_frame(run_dirs, baseline_dirs)

    # Resolve output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_run_directory(base_dir=results_dir, prefix='report')

    print(f"Writing report to {output_dir}")

    # Master CSV
    master_csv = os.path.join(output_dir, 'master_grid_comparison.csv')
    df.to_csv(master_csv, index=False)
    print(f"  wrote {master_csv} ({len(df)} rows, {len(df.columns)} cols)")

    # Heatmaps (only for non-baseline runs to keep the grid clean).
    # When multiple margin sets are present, emit one heatmap per (margin, metric)
    # pair so each margin table's grid surface is visible separately.
    heatmap_paths: List[Tuple[str, str]] = []
    if not df.empty:
        grid_only = df[df['is_baseline'] == False].copy()  # noqa: E712
        heatmap_dir = os.path.join(output_dir, 'heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)

        margin_groups: List[Tuple[str, pd.DataFrame]] = []
        if 'margin_tag' in grid_only.columns:
            tags = grid_only['margin_tag'].fillna('default')
            unique_tags = sorted(tags.unique().tolist())
            if len(unique_tags) > 1:
                for tag in unique_tags:
                    sub = grid_only[tags == tag]
                    margin_groups.append((tag, sub))
            else:
                margin_groups.append((unique_tags[0] if unique_tags else 'all',
                                      grid_only))
        else:
            margin_groups.append(('all', grid_only))

        for tag, sub_df in margin_groups:
            for metric, ablation, label in metrics:
                if metric not in df.columns:
                    continue
                tag_suffix = f"_{tag}" if tag not in ('all',) else ''
                fname = f"heatmap_{ablation}_{metric}{tag_suffix}.png"
                full_path = os.path.join(heatmap_dir, fname)
                tag_label = f"{label} [margins={tag}]" if tag != 'all' else label
                written = _generate_heatmap(sub_df, metric, ablation,
                                            tag_label, full_path)
                if written:
                    heatmap_paths.append(
                        (tag_label, os.path.join('heatmaps', fname)))
        print(f"  wrote {len(heatmap_paths)} heatmap(s) "
              f"across {len(margin_groups)} margin group(s)")

    # Markdown report
    md_path = _write_report(df, incomplete, metrics, args.top_n,
                            heatmap_paths, output_dir, results_dir)
    print(f"  wrote {md_path}")

    # JSON companion
    json_path = os.path.join(output_dir, 'grid_report.json')
    json_payload = {
        'results_dir': results_dir,
        'n_runs_completed': int(df['run_name'].nunique()) if not df.empty else 0,
        'n_runs_incomplete': len(incomplete),
        'metrics_headlined': [
            {'metric': m, 'ablation': a, 'label': l} for (m, a, l) in metrics],
        'incomplete_runs': incomplete,
        'cw_values': sorted(df['contrastive_weight'].dropna().unique().tolist())
                     if not df.empty else [],
        'hw_values': sorted(df['head_weight'].dropna().unique().tolist())
                     if not df.empty else [],
    }
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, indent=2, default=str)
    print(f"  wrote {json_path}")

    # README manifest
    extra_files = {
        'master_grid_comparison.csv': 'tidy long-format master table; one row per (run, ablation_config)',
        'grid_report.md': 'human-readable comparison report',
        'grid_report.json': 'machine-readable summary of the report',
        'heatmaps/': f'{len(heatmap_paths)} cw × hw heatmaps (PNG)',
    }
    write_run_manifest(
        output_dir,
        description=f"Grid search comparison report aggregated from {results_dir}",
        approach="Aggregation across hierarchy_grid_* run dirs",
        script_path=os.path.abspath(__file__),
        notes=(f"{int(df['run_name'].nunique()) if not df.empty else 0} completed run(s), "
               f"{len(incomplete)} incomplete. Re-run this script as more grid jobs finish."),
        extra_files=extra_files,
    )

    print(f"\nDone. Report: {md_path}")


if __name__ == '__main__':
    main()
