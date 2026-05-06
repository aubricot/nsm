# `hierarchy/tests/` — pytest suite

Unit tests for the `hierarchy` package. Every test file targets a single sibling module in `hierarchy/`.

## Files

| Test file | Module under test | What it covers |
|---|---|---|
| `test_taxonomy.py` | `hierarchy/taxonomy.py` | Filename → taxonomy parsing, `TaxonomyTree` construction, `HierarchicalMultiTaskClassifier` forward pass |
| `test_losses.py` | `hierarchy/losses.py` | `TaxonomyLabelEncoder` label encoding, `HierarchyContrastiveLoss` per-distance margin behavior, `TaxonomyClassificationHeads` shape, `compute_classification_head_loss` masking of missing labels |
| `test_classifiers.py` | `hierarchy/classifiers.py` | sklearn classifier wrappers (KNN/SVM/RF/MLP/LR), MultiOutputClassifier with 2D targets, position regressors |
| `test_metric_learning.py` | `hierarchy/metric_learning.py` | `LatentMetricLearner` fit + transform with NCA, idempotence, dimensionality preservation |
| `test_evaluation.py` | `hierarchy/evaluation.py` | `calculate_metrics` (top-k, macro/weighted F1), hierarchical confusion matrices, regression MAE/R² |
| `test_spine_position.py` | `hierarchy/spine_position.py` | `SpinePositionMapper` filename parsing, normalized position formula, derived-region boundaries |

## Run

From the project root (parent of `hierarchy/`):

```bash
# Full suite
pytest hierarchy/tests/

# A single file
pytest hierarchy/tests/test_losses.py -v

# A single test by name
pytest hierarchy/tests/test_evaluation.py::test_top_k_accuracy -v
```

Tests use pytest's standard discovery (`test_*.py` + `test_*` functions). No conftest, no fixtures shared across files.

## Known caveats

- A handful of tests that use `torch.cdist` may segfault on macOS (Apple-Silicon MPS backend); this is an upstream PyTorch issue, not a defect in the code being tested. Run on CUDA or skip with `pytest -k "not cdist"` if you hit it.
- Tests don't touch the filesystem outside `tmp_path` and don't require the `vertebrae_meshes/` data dir — they're pure unit tests on synthetic inputs.
