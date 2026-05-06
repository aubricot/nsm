# `hierarchy/run/` — CLI entry-point shims

Each file here is a thin wrapper that imports `main()` from a sibling module in the `hierarchy/` package and calls it. The shims exist so that the implementation modules stay importable as a normal Python package (with relative imports) while still being invokable as scripts.

## Files

```
run/
├── __init__.py
├── run_train_hierarchy.py     →  hierarchy.train.main
├── run_classify.py            →  hierarchy.classify.main
└── run_grid_report.py         →  hierarchy.grid_report.main
```

Each file is two lines: `from hierarchy.X import main` then `main()`.

## How to run

From the project root (i.e. the parent of `hierarchy/`):

```bash
# Train a hierarchy-aware model
python hierarchy/run/run_train_hierarchy.py [args...]

# Classify a trained run
python hierarchy/run/run_classify.py --run-dir hierarchy_v1

# Aggregate grid-search runs into a comparison report
python hierarchy/run/run_grid_report.py
```

Use `--help` on any of them to see all flags.

## Why a separate directory

If `run_*.py` lived at the project root they'd clutter the top level alongside unrelated scripts. If they lived inside `hierarchy/` next to `train.py` etc., the package's relative imports (`from .losses import …`) would conflict with running the file directly. Keeping them in a sibling subdirectory under `hierarchy/run/` solves both: they're co-located with the package they invoke, but importing `hierarchy.train` from here works because the project root is on `sys.path` when you run `python hierarchy/run/run_train_hierarchy.py`.

## Adding a new entry point

1. Add a `main()` function (with its own argparse) to a module in `hierarchy/`.
2. Create a two-line shim in this directory:
   ```python
   from hierarchy.your_module import main
   main()
   ```
3. Document it in this README and the package README.
