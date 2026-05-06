# `hierarchy/` — taxonomy-aware NSM training & classification

Adds taxonomic structure to the upstream Neural Shape Model:

1. **Hierarchy-aware contrastive loss** on the latent code with per-taxonomic-distance margins (same-species pairs attract; same-genus / same-family / different-family pairs repel with progressively larger margins).
2. **Multi-head classification** that predicts species, genus, family, and spinal position directly from the latent.
3. **Ablation classification pipeline** (raw retrieval → metric learning → supervised classifiers → both) with hierarchical evaluation at every taxonomic level.

The upstream `NSM/` package is **unchanged**. Everything in this directory composes on top of it.

## File structure

```
hierarchy/
├── taxonomy.py           # parse_taxonomy_from_filename, TaxonomyTree, HierarchicalMultiTaskClassifier
├── losses.py             # HierarchyContrastiveLoss, TaxonomyClassificationHeads, TaxonomyLabelEncoder
├── train.py              # Training loop = NSM SDF loss + hierarchy contrastive + multi-head CE
├── classify.py           # Ablation pipeline: A baseline → B metric-learning → C classifiers → D both
├── classifiers.py        # sklearn classifiers + regressors (KNN, SVM, RF, MLP, LR; KNN/SVR/RF/MLP regressors)
├── metric_learning.py    # LatentMetricLearner — NCA/LMNN wrapper for re-shaping the latent space
├── evaluation.py         # Top-k accuracy, F1, hierarchical confusion matrices, regression MAE/R²
├── spine_position.py     # SpinePositionMapper — continuous normalized position from vertebrae_counts.csv
├── interpretation.py     # Logistic-regression coefficient analysis (which latent dims drive species)
├── reports.py            # Single-run summary stats + plots (used by classify.py)
├── grid_report.py        # Cross-run aggregation across grid_search_results/ (used by run_grid_report.py)
├── run_utils.py          # create_run_directory (timestamped, non-overwriting), write_run_manifest
├── visualize.py          # PCA / t-SNE / UMAP projections of latents (standalone script)
├── run/                  # CLI entry-point shims — see run/README.md
└── tests/                # pytest suite — see tests/README.md
```

## How to use

All commands assume you are at the project root (the parent of `hierarchy/`) and have the `NSM` conda env active.

### Train a hierarchy-aware model
```bash
python hierarchy/run/run_train_hierarchy.py \
    --contrastive-weight 0.01 \
    --head-weight 0.005 \
    --contrastive-margins 0.0 1.0 2.0 4.0
```
- Run dir is auto-named `hierarchy_v<N>` (or whatever you pass to `--run-name`).
- The full hyperparameter set is dumped to `<run_dir>/hyperparams.json` *before* training starts, so a crashed run still leaves a record.
- See `python hierarchy/run/run_train_hierarchy.py --help` for every knob.

### Run the classify ablation pipeline against a trained run
```bash
python hierarchy/run/run_classify.py --run-dir hierarchy_v1
```
Produces `<run_dir>/results/ablation_*/` with `comparison_metrics.csv`, per-cell predictions, confusion matrices, and a markdown summary.

### Aggregate a grid search into a comparison report
```bash
python hierarchy/run/run_grid_report.py --include-baselines
```
Reads every `hierarchy_grid_*` dir under `hierarchy/grid_search_results/`, writes a fresh `report_<timestamp>/` containing `master_grid_comparison.csv`, `grid_report.md`, and per-margin-set heatmaps.

### Run the tests
```bash
pytest hierarchy/tests/
```

## Public API

`from hierarchy import …` re-exports the most commonly imported objects (see `__init__.py`):
`parse_taxonomy_from_filename`, `TaxonomyTree`, `TaxonomyLabelEncoder`, `HierarchyContrastiveLoss`, `TaxonomyClassificationHeads`, `compute_classification_head_loss`, `LatentMetricLearner`, `SpinePositionMapper`, `calculate_metrics`, `create_run_directory`, etc.

## Required external data

- `vertebrae_meshes/*.vtk` — per-vertebra meshes (filename encodes taxonomy + position)
- `vertebrae_config.json` — base NSM training config (latent dim, sampling, etc.)
- `vertebrae_counts.csv` — *optional*; enables continuous normalized spine position. Without it, classify falls back to discrete 3-class (Cervical / Thoracic / Lumbar) only.
