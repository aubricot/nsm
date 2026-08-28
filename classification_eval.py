"""
validation metrics for latent space vertebra classification.

classification here is nearest neighbour retrieval in the learned latent space:
a query vertebra gets the labels of the training vertebra whose latent is most
similar (cosine). this measures how well the latent space separates taxonomy,
spinal region and spinal position, and writes the precision/recall/f1 tables and
confusion matrices for the paper.

eval levels (what gets hidden when a vertebra is classified):
    loo       only the query itself. same specimen neighbours allowed, so
              taxonomy is an optimistic upper bound (specimen re identification).
    specimen  hide the whole specimen. honest "classify a new individual" test.
    species   hide every specimen of the same species.
    genus     hide every specimen of the same genus.

categories:
    family, genus, species   taxonomy, dibamus dropped (no valid higher taxon)
    broad_taxon              higher grouping for phylogeny aware plots, so sister
                             families read as correct without knowing the tree
    region                   cervical / thoracic / lumbar, amphisbaenians dropped
    position_10, position_20 normalized position binned into 10% and 20% chunks
    life_history             ecomorph from the species list trait column

outputs land under <run_name>/classification_eval/<suffix>/.
"""

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

REGION_NAMES = {"c": "Cervical", "t": "Thoracic", "l": "Lumbar"}
REGION_ORDER = ["Cervical", "Thoracic", "Lumbar"]


# helpers
def _norm(name):
    # lowercase, drop extension, collapse runs of dash and underscore to one underscore
    return re.sub(r"[-_]+", "_", str(name).lower().rsplit(".", 1)[0])


def _digits(text):
    # accession number, the last <letters><3+ digits> run. ignores short ordinals
    m = re.findall(r"[a-z]+-?(\d{3,})", str(text).lower())
    return m[-1] if m else None


def _region_num(fname):
    m = re.search(r"-([ctlCTL])(\d+)\.[^.]+$", fname)
    return (m.group(1).lower(), int(m.group(2))) if m else (None, None)


def load_species_meta(path):
    # per specimen metadata keyed a few ways so messy specimen strings still match.
    # cham in the mesh names is the abbreviation for chamaesaura.
    df = pd.read_csv(path)
    df["g"] = df["genus"].astype(str).str.lower()
    df["s"] = df["species"].astype(str).str.lower()

    by_norm, by_gd, by_gs = {}, {}, {}
    for r in df.itertuples():
        meta = {
            "broad_taxon": (str(r.broad_taxon_for_plotting).lower()
                            if pd.notna(r.broad_taxon_for_plotting) else None),
            "life_history": (str(r.trait).strip().lower()
                             if pd.notna(r.trait) else None),
        }
        by_norm[_norm(r.specimen)] = meta
        d = _digits(r.specimen)
        if d:
            by_gd[(r.g, d)] = meta
        by_gs[(r.g, r.s)] = meta

    def lookup(specimen, genus, species):
        g = (genus or "").lower()
        if g == "cham":
            g = "chamaesaura"
        n = _norm(specimen)
        if n in by_norm:
            return by_norm[n]
        d = _digits(specimen)
        if d and (g, d) in by_gd:
            return by_gd[(g, d)]
        return by_gs.get((g, (species or "").lower()), {"broad_taxon": None,
                                                        "life_history": None})
    return lookup


def build_labels(train_files, mapping_csv, species_meta):
    # three tier join to the curated per vertebra mapping: exact normalized name,
    # then specimen level recovery by accession digits, then a filename token parse.
    df = pd.read_csv(mapping_csv)
    df["fam"] = df["family"].str.lower()
    df["reg"] = df["vertebra_type"].str.lower()
    by_norm = {_norm(r.vtk_name): r for r in df.itertuples()}

    spec_tax, spec_pos = {}, {}
    for r in df.itertuples():
        d = _digits(r.specimen)
        if d:
            spec_tax.setdefault((r.fam, d), (r.family, r.genus, r.species, r.specimen))
            spec_pos[(d, r.reg, r.vertebra_num)] = r.normalized_position

    rows = []
    for f in train_files:
        n = _norm(f)
        reg, num = _region_num(f)
        family = genus = species = specimen = None
        norm_pos = np.nan
        source = "unmatched"

        if n in by_norm:
            r = by_norm[n]
            family, genus, species, specimen = r.family, r.genus, r.species, r.specimen
            reg = reg or r.reg
            num = num if num is not None else r.vertebra_num
            norm_pos = r.normalized_position
            source = "csv_exact"
        else:
            fam0 = f.split("_")[0].split("-")[0].lower()
            d = _digits(f.rsplit(".", 1)[0])
            if d and (fam0, d) in spec_tax:
                family, genus, species, specimen = spec_tax[(fam0, d)]
                norm_pos = spec_pos.get((d, reg, num), np.nan)
                source = "csv_specimen"
            else:
                toks = re.split(r"[-_]", f.rsplit(".", 1)[0])
                family = toks[0] if toks else None
                genus = toks[1] if len(toks) > 1 else None
                species = toks[2] if len(toks) > 2 else None
                specimen = f"{family}_{genus}_{species}_{d or 'na'}"
                source = "filename"

        meta = species_meta(specimen, genus, species)
        rows.append({
            "mesh": f,
            "family": (family or "").lower() or None,
            "genus": (genus or "").lower() or None,
            "species": (f"{genus}_{species}".lower() if genus and species else None),
            "broad_taxon": meta["broad_taxon"],
            "region": REGION_NAMES.get(reg),
            "norm_pos": norm_pos,
            "life_history": meta["life_history"],
            "specimen": (specimen or "").lower() or None,
            "label_source": source,
        })
    return pd.DataFrame(rows)


def rank_neighbours(latents, group_ids, level, topk):
    # for each row, indices of the most similar other rows by cosine.
    # under a leave one group out level, rows in the query's group are masked.
    latents = F.normalize(latents.float(), dim=1)
    sim = latents @ latents.t()
    n = sim.shape[0]
    sim[torch.arange(n), torch.arange(n)] = -2.0
    if level != "loo":
        g = np.asarray(group_ids, dtype=object)
        sim[torch.from_numpy(g[:, None] == g[None, :])] = -2.0
    return torch.argsort(sim, dim=1, descending=True)[:, :topk].cpu().numpy()


def evaluate_category(labels, order, eligible=None, k_top5=5):
    # top 1 / top 5 predictions and correctness for one label column.
    # eligible marks queries where a correct answer is possible at all, so under
    # leave one group out a singleton class does not drag the score down unfairly.
    y = labels.to_numpy(dtype=object)
    valid = np.array([v is not None and not (isinstance(v, float) and np.isnan(v))
                      for v in y])
    if eligible is None:
        eligible = np.ones(len(y), dtype=bool)

    top1, top5, keep = [], [], []
    for i in range(len(y)):
        if not valid[i]:
            keep.append(False); top1.append(None); top5.append(False); continue
        neigh = [j for j in order[i] if valid[j]]
        if not neigh:
            keep.append(False); top1.append(None); top5.append(False); continue
        keep.append(True)
        top1.append(y[neigh[0]])
        top5.append(y[i] in {y[j] for j in neigh[:k_top5]})

    keep = np.array(keep)
    yt = y[keep]
    yp = np.array(top1, dtype=object)[keep]
    t5 = np.array(top5)[keep]
    el = eligible[keep]

    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    report = classification_report(yt, yp, zero_division=0, output_dict=True)
    summary = {
        "n_eval": int(keep.sum()),
        "n_classes": int(len(set(yt))),
        "top1_accuracy": float((yt == yp).mean()),
        "top5_accuracy": float(t5.mean()),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "weighted_precision": float(pw), "weighted_recall": float(rw), "weighted_f1": float(f1w),
        "n_eligible": int(el.sum()),
        "n_reachable_classes": int(len(set(yt[el]))),
        "top1_accuracy_reachable": float((yt[el] == yp[el]).mean()) if el.any() else float("nan"),
        "top5_accuracy_reachable": float(t5[el].mean()) if el.any() else float("nan"),
    }
    return summary, report, yt, yp, np.array(top1, dtype=object), np.array(top5)


def plot_confusion(y_true, y_pred, title, out_png, class_order=None, normalize=True):
    present = set(y_true) | set(y_pred)
    if class_order:
        classes = [c for c in class_order if c in present]
    else:
        classes = sorted(present)
    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    size = max(4, 0.45 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row normalised fraction" if normalize else "count")
    if len(classes) <= 15:
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2f}" if normalize else int(v), ha="center",
                            va="center", fontsize=6, color="white" if v < 0.6 else "black")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _load_latents(run_dir, checkpoint):
    obj = torch.load(os.path.join(run_dir, "latent_codes", f"{checkpoint}.pth"),
                     map_location="cpu")
    lc = obj["latent_codes"] if isinstance(obj, dict) and "latent_codes" in obj else obj
    if isinstance(lc, dict) and "weight" in lc:
        lc = lc["weight"]
    if hasattr(lc, "weight"):
        lc = lc.weight
    return lc.detach().float()


def _latest_checkpoint(run_dir):
    d = os.path.join(run_dir, "latent_codes")
    ckpts = [int(os.path.splitext(f)[0]) for f in os.listdir(d)
             if f.endswith(".pth") and os.path.splitext(f)[0].isdigit()]
    if not ckpts:
        raise FileNotFoundError(f"no numeric checkpoints in {d}")
    return str(max(ckpts))


def parse_args():
    p = argparse.ArgumentParser(description="latent space classification validation metrics")
    p.add_argument("--run_name", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mapping_csv", default="vtk_name_to_mapping_v2.csv")
    p.add_argument("--species_list", default="lizard_species_list.csv")
    p.add_argument("--eval_level", choices=["loo", "specimen", "species", "genus"],
                   default="specimen")
    p.add_argument("--cm_max_classes", type=int, default=40)
    p.add_argument("--suffix", default=None)
    return p.parse_args()


def bin_positions(values, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    labels_order = [f"{edges[i]*100:.0f}-{edges[i+1]*100:.0f}%" for i in range(n_bins)]
    def to_bin(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        idx = min(int(np.digitize(v, edges[1:-1])), n_bins - 1)
        return labels_order[idx]
    return values.map(to_bin), labels_order


def main():
    args = parse_args()
    run_dir = args.run_name
    ckpt = args.checkpoint or _latest_checkpoint(run_dir)
    suffix = args.suffix or f"{args.eval_level}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

    config = json.load(open(os.path.join(run_dir, "model_params_config.json")))
    train_files = [os.path.basename(p) for p in config["list_mesh_paths"]]

    latents = _load_latents(run_dir, ckpt)
    assert latents.shape[0] == len(train_files), \
        f"latent count {latents.shape[0]} != file count {len(train_files)}"

    species_meta = load_species_meta(args.species_list)
    labels = build_labels(train_files, args.mapping_csv, species_meta)

    labels["position_10"], order_10 = bin_positions(labels["norm_pos"], 10)
    labels["position_20"], order_20 = bin_positions(labels["norm_pos"], 5)

    # exclusions katherine asked for. dibamus has no valid higher taxon, and
    # amphisbaenian regions are not distinguishable during segmentation.
    is_dibamus = labels["genus"] == "dibamus"
    is_amph = labels["broad_taxon"] == "amphisbaenea"
    for cat in ["family", "genus", "species", "broad_taxon"]:
        labels.loc[is_dibamus, cat] = None
    labels.loc[is_amph, "region"] = None

    out_dir = os.path.join(run_dir, "classification_eval", suffix)
    os.makedirs(out_dir, exist_ok=True)
    labels.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    group_col = {"specimen": "specimen", "species": "species", "genus": "genus"}.get(
        args.eval_level)
    group_ids = (labels[group_col].to_numpy() if group_col
                 else np.arange(len(labels)))
    order = rank_neighbours(latents, group_ids, args.eval_level, topk=5)

    cats = {
        "family": None, "genus": None, "species": None, "broad_taxon": None,
        "region": REGION_ORDER, "position_10": order_10, "position_20": order_20,
        "life_history": None,
    }

    def eligibility(cat):
        if args.eval_level == "loo":
            return None
        n = labels.groupby(cat)[group_col].transform("nunique")
        return (n >= 2).to_numpy()

    summary_rows, all_json = [], {
        "run": run_dir, "checkpoint": ckpt, "eval_level": args.eval_level,
        "n_train": len(train_files), "categories": {},
    }
    preds = labels[["mesh", "specimen"]].copy()

    print(f"\neval_level: {args.eval_level}")
    print(f"{'category':<14}{'n':>6}{'cls':>5}{'top1':>8}{'top5':>8}"
          f"{'macroF1':>9}{'reach_top1':>11}{'reach_cls':>10}")
    print("-" * 71)
    for cat, corder in cats.items():
        summ, report, yt, yp, full1, full5 = evaluate_category(
            labels[cat], order, eligible=eligibility(cat))
        summ["category"] = cat
        summary_rows.append(summ)
        all_json["categories"][cat] = {"summary": summ, "per_class": report}
        preds[f"{cat}_true"] = labels[cat]
        preds[f"{cat}_pred"] = full1
        preds[f"{cat}_top5_hit"] = full5
        pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, f"report_{cat}.csv"))
        if summ["n_classes"] <= args.cm_max_classes:
            plot_confusion(yt, yp,
                           f"{cat}  ({args.eval_level}, top1={summ['top1_accuracy']:.2f})",
                           os.path.join(out_dir, f"confusion_{cat}.png"), class_order=corder)
        print(f"{cat:<14}{summ['n_eval']:>6}{summ['n_classes']:>5}"
              f"{summ['top1_accuracy']:>8.3f}{summ['top5_accuracy']:>8.3f}"
              f"{summ['macro_f1']:>9.3f}{summ['top1_accuracy_reachable']:>11.3f}"
              f"{summ['n_reachable_classes']:>10}")

    pd.DataFrame(summary_rows).set_index("category").to_csv(
        os.path.join(out_dir, "metrics_summary.csv"))
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    json.dump(all_json, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    print(f"\nlabel sources: {labels['label_source'].value_counts().to_dict()}")
    print(f"broad_taxon coverage: {labels['broad_taxon'].notna().sum()}/{len(labels)}")
    print(f"life_history coverage: {labels['life_history'].notna().sum()}/{len(labels)}")
    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
