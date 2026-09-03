"""
validation metrics for latent space vertebra classification.

classification here is nearest neighbour retrieval in the learned latent space:
a query vertebra (e.g., from validation set) gets the labels of the gallery vertebra
(e.g., from the training set) whose latent is most similar (cosine). 

eval levels (what gets hidden when a vertebra is classified):
    loo       hide the exact vertebra (optimistic upper bound).
    specimen  hide the whole specimen (all vertebrae for one specimen). honest "classify a new individual" test.
    species   hide every specimen of the same species.
    genus     hide every specimen of the same genus.

# Example usage:
#   conda activate NSM
#   cd NSM/nsm

#   Train set, leave-one-out (LOO):
#   python classify_latents.py \
#       --model_root run_v72 \
#       --ckpt 2500 \
#       --dataset_split train \
#       --eval_level loo 
#
#   Train set, LOSO (specimen-level exclusion):
#   python classify_latents.py \
#       --model_root run_v72 \
#       --ckpt 2500 \
#       --dataset_split train \
#       --eval_level specimen
#
#   Train set using precomputed encoded latents via optimization (from encode_latents_for_eval.py):
#   python classify_latents.py \
#       --model_root run_v72 \
#       --ckpt 2500 \
#       --dataset_split val \
#       --encoded_latents \
#       --eval_level specimen

"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from NSM.helper_funcs import load_config, load_model_and_latents

REGION_NAMES = {"c": "Cervical", "t": "Thoracic", "l": "Lumbar"}
REGION_ORDER = ["Cervical", "Thoracic", "Lumbar"]

PAT = re.compile(r"^(?P<specimen>[\w\s\-]+)[\-_ ]+[\w\d]+[\-_ ]+(?P<vertebra>[CTL]\d+)",re.IGNORECASE)

def parse_mesh_info(text):
    text_str = str(text).lower()
    m_pat    = PAT.match(text_str)
    m_region = re.search(r"_(\d+)-([ctl])(\d+)\.[^.]+$", text_str)
    m_digits = re.findall(r"[a-z]+-?(\d{3,})", text_str)

    if m_pat:
        specimen = m_pat.group("specimen").strip().replace("-", "_")
    else:
        # fallback for files with no region suffix e.g. amphisbaenidae_amphisbaena_bolivica_uf141550_002.vtk
        specimen = re.sub(r"_\d+\.[^.]+$", "", text_str).replace("-", "_")

    return {"specimen": specimen,
            "spec_id":  m_digits[-1] if m_digits else None,
            "seq_num":  int(m_region.group(1)) if m_region else None,
            "region":   m_region.group(2)      if m_region else None,
            "num":      int(m_region.group(3)) if m_region else None,}

def compute_norm_pos(seq_num, total_vert):
    try:
        return int(seq_num) / int(total_vert)
    except (ValueError, TypeError):
        return np.nan

def load_species_meta(path):
    df = pd.read_csv(path)
    meta = {}
    for r in df.itertuples():
        key = str(r.specimen).lower().strip().replace("-", "_")
        meta[key] = {
            "broad_taxon":  str(r.broad_taxon_for_plotting).lower() if pd.notna(r.broad_taxon_for_plotting) else None,
            "life_history": str(r.trait).strip().lower()            if pd.notna(r.trait) else None,
            "family":       str(r.family).lower()                   if pd.notna(r.family) else None,
            "genus":        str(r.genus).lower()                    if pd.notna(r.genus) else None,
            "species":      f"{r.genus}_{r.species}".lower()        if pd.notna(r.genus) and pd.notna(r.species) else None,
            "total_vert":   r.total_vert                            if pd.notna(r.total_vert) else None,
        }
    return meta

def check_labels(labels, name="labels"):
    for col in ["specimen", "family", "species", "region"]:
        pct = 1 - labels[col].notna().mean()
        print(f"[{name}] {col:<9} Data not matched: {pct:6.1%}")
        if pct < 1:
            bad = labels.loc[labels[col].isna(), "mesh"].head(3).tolist()
            print(f"missing e.g. {bad}")
    return labels

def build_labels(file_list, species_meta, name="labels"):
    rows = []
    for f in file_list:
        info  = parse_mesh_info(f)
        spec_key = info["specimen"]
        meta  = species_meta.get(spec_key, {})
        try:
            norm_pos = info["seq_num"] / int(meta["total_vert"]) if info["seq_num"] and meta.get("total_vert") else np.nan
        except (ValueError, TypeError):
            norm_pos = np.nan
        rows.append({"mesh":         f,
                     "family":       meta.get("family"),
                     "genus":        meta.get("genus"),
                     "species":      meta.get("species"),
                     "specimen":     spec_key,
                     "broad_taxon":  meta.get("broad_taxon"),
                     "region":       REGION_NAMES.get(info["region"]),
                     "norm_pos":     norm_pos,
                     "life_history": meta.get("life_history")})
    return check_labels(pd.DataFrame(rows), name=name)

def get_eligibility(y_q, y_g, q_hide, g_hide):
    y_q = np.asarray(y_q, dtype=object)
    y_g = np.asarray(y_g, dtype=object)
    q_h = np.asarray(q_hide, dtype=object)
    g_h = np.asarray(g_hide, dtype=object)
    
    eligible = []
    for i in range(len(y_q)):
        if pd.isna(y_q[i]):
            eligible.append(False)
            continue
        q_val = q_h[i]
        allowed = np.ones(len(g_h), dtype=bool) if pd.isna(q_val) else (g_h != q_val) | pd.isna(g_h)
        eligible.append(y_q[i] in set(y_g[allowed]))
    return np.array(eligible, dtype=bool)

def rank_neighbours(q_latents, g_latents, q_hide, g_hide, topk, mask=True):
    q_latents = F.normalize(q_latents.float(), dim=1)
    g_latents = F.normalize(g_latents.float(), dim=1)
    sim = q_latents @ g_latents.t()

    if mask:
        q_h = np.asarray(q_hide, dtype=object)
        g_h = np.asarray(g_hide, dtype=object)
        valid_q = np.array([pd.notna(v) for v in q_h])
        valid_g = np.array([pd.notna(v) for v in g_h])
        match_matrix = (q_h[:, None] == g_h[None, :])
        valid_matrix = valid_q[:, None] & valid_g[None, :]
        sim[torch.from_numpy(match_matrix & valid_matrix)] = -2.0
    return torch.argsort(sim, dim=1, descending=True)[:, :topk].cpu().numpy()

def evaluate_category(y_q, y_g, order, eligible=None, k_top5=5):
    y_q = y_q.to_numpy(dtype=object)
    y_g = y_g.to_numpy(dtype=object)
    valid_q = np.array([pd.notna(v) for v in y_q])
    valid_g = np.array([pd.notna(v) for v in y_g])
    if eligible is None:
        eligible = np.ones(len(y_q), dtype=bool)

    top1, top5, keep = [], [], []
    for i in range(len(y_q)):
        if not valid_q[i]:
            keep.append(False); top1.append(None); top5.append(False); continue
        neigh = [j for j in order[i] if valid_g[j]]
        if not neigh:
            keep.append(False); top1.append(None); top5.append(False); continue
        keep.append(True)
        top1.append(y_g[neigh[0]])
        top5.append(y_q[i] in {y_g[j] for j in neigh[:k_top5]})

    keep = np.array(keep)
    yt = y_q[keep]
    yp = np.array(top1, dtype=object)[keep]
    t5 = np.array(top5)[keep]
    el = eligible[keep]

    p, r, f1, _   = precision_recall_fscore_support(yt, yp, average="macro",    zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    report = classification_report(yt, yp, zero_division=0, output_dict=True)
    summary = {"n_eval": int(keep.sum()),
               "n_classes": int(len(set(yt))),
               "top1_accuracy": float((yt == yp).mean()),
               "top5_accuracy": float(t5.mean()),
               "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
               "weighted_precision": float(pw), "weighted_recall": float(rw), "weighted_f1": float(f1w),
               "n_eligible": int(el.sum()),
               "n_reachable_classes": int(len(set(yt[el]))),
               "top1_accuracy_reachable": float((yt[el] == yp[el]).mean()) if el.any() else float("nan"),
               "top5_accuracy_reachable": float(t5[el].mean()) if el.any() else float("nan")}
    return summary, report, yt, yp, np.array(top1, dtype=object), np.array(top5)

def plot_confusion(y_true, y_pred, title, out_png, class_order=None, normalize=True):
    present = set(y_true) | set(y_pred)
    classes = [c for c in class_order if c in present] if class_order else sorted(present)
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
    ax.set_xlabel("predicted (Gallery)"); ax.set_ylabel("true (Query)"); ax.set_title(title)
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

def bin_positions(values, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    labels_order = [f"{edges[i]*100:.0f}-{edges[i+1]*100:.0f}%" for i in range(n_bins)]
    def to_bin(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        idx = min(int(np.digitize(v, edges[1:-1])), n_bins - 1)
        return labels_order[idx]
    return values.map(to_bin), labels_order

def apply_exclusions(labels):
    is_dibamus = labels["genus"] == "dibamus"
    is_amph    = labels["broad_taxon"] == "amphisbaenea"
    is_rhyncho = labels["genus"] == "sphenodon"
    is_snake   = labels["broad_taxon"] == "snake"
    for cat in ["family", "genus", "species", "broad_taxon"]:
        labels.loc[is_dibamus | is_rhyncho | is_snake, cat] = None
    labels.loc[is_amph, "region"] = None
    return labels

def main():
    args = parse_args()
    run_dir = args.model_root
    ckpt = args.ckpt
    config = load_config(config_path=f"{run_dir}/model_params_config.json")
    device = config.get("device", "cuda:0")

    eval_tag = f'{args.eval_level}_{"base" if not args.encoded_latents else "latent_opt"}'
    suffix   = args.suffix or f"{args.dataset_split}_{eval_tag}"
    out_dir  = os.path.join(run_dir, "classification/evaluation", args.dataset_split, suffix)

    ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
    gallery_files = [os.path.basename(p) for p in config["list_mesh_paths"]]
    if args.encoded_latents:
        query_files = [os.path.basename(p) for p in config[ds_split_keys[args.dataset_split]]]
    else:
        query_files = gallery_files  # train LOO/specimen/etc

    print(f"Loading gallery latents from: {args.gallery_latents}")
    model_path = os.path.join(run_dir, "model", f"{ckpt}.pth")
    _, _, gallery_latents = load_model_and_latents(model_path, args.gallery_latents, config, device)
    gallery_latents = gallery_latents.to(device)

    if args.encoded_latents:
        print(f"Loading query latents from: {args.query_latents}")
        query_latents = torch.load(args.query_latents, map_location=device, weights_only=True).to(device)
    else:
        query_latents = gallery_latents  # same set, eval_level masking handles exclusion

    if gallery_latents.shape[0] != len(gallery_files):
        raise ValueError(f"Gallery latents {gallery_latents.shape} != {len(gallery_files)} files.")
    if query_latents.shape[0] != len(query_files):
        raise ValueError(f"Query latents {query_latents.shape} != {len(query_files)} files.")

    species_meta = load_species_meta(args.species_list)
    labels_g = build_labels(gallery_files, species_meta, "gallery")
    labels_q = build_labels(query_files,   species_meta, "query")

    labels_g["position_10"], order_10 = bin_positions(labels_g["norm_pos"], 10)
    labels_g["position_20"], order_20 = bin_positions(labels_g["norm_pos"], 5)
    labels_q["position_10"], _        = bin_positions(labels_q["norm_pos"], 10)
    labels_q["position_20"], _        = bin_positions(labels_q["norm_pos"], 5)

    labels_g = apply_exclusions(labels_g)
    labels_q = apply_exclusions(labels_q)

    os.makedirs(out_dir, exist_ok=True)
    labels_q.to_csv(os.path.join(out_dir, "query_labels.csv"), index=False)

    mask     = True  #not args.encoded_latents ## TO DO: KW changed for testing
    group_col = "mesh" if args.eval_level == "loo" else args.eval_level
    g_hide   = labels_g[group_col].to_numpy()
    q_hide   = labels_q[group_col].to_numpy()
    order    = rank_neighbours(query_latents, gallery_latents, q_hide, g_hide, topk=50, mask=mask)

    cats = {"family": None, "genus": None, "species": None, "broad_taxon": None,
            "region": REGION_ORDER, "position_10": order_10, "position_20": order_20,
            "life_history": None}
    summary_rows = []
    all_json = {"run": run_dir, "ckpt": ckpt,
                "eval_level": args.eval_level,
                "n_query": len(query_files), "n_gallery": len(gallery_files), "categories": {}}
    preds = labels_q[["mesh", "specimen"]].copy()

    print(f"\nQuery: {args.dataset_split} ({len(query_files)}) | Gallery: train ({len(gallery_files)})")
    print(f"eval_level: {args.eval_level}")
    print(f"{'category':<14}{'n':>6}{'cls':>5}{'top1':>8}{'top5':>8}{'macroF1':>9}{'reach_top1':>11}{'reach_cls':>10}")
    print("-" * 71)

    for cat, corder in cats.items():
        eligible = get_eligibility(labels_q[cat], labels_g[cat], q_hide, g_hide)
        summ, report, yt, yp, full1, full5 = evaluate_category(labels_q[cat], labels_g[cat], order, eligible=eligible)
        summ["category"] = cat
        summary_rows.append(summ)
        all_json["categories"][cat] = {"summary": summ, "per_class": report}
        preds[f"{cat}_true"] = labels_q[cat]
        preds[f"{cat}_pred"] = full1
        preds[f"{cat}_top5_hit"] = full5
        pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, f"report_{cat}.csv"))
        if summ["n_classes"] <= args.cm_max_classes:
            plot_confusion(yt, yp, f"{cat}  ({args.eval_level}, top1={summ['top1_accuracy']:.2f})",
                           os.path.join(out_dir, f"confusion_{cat}.png"), class_order=corder)
        print(f"{cat:<14}{summ['n_eval']:>6}{summ['n_classes']:>5}"
              f"{summ['top1_accuracy']:>8.3f}{summ['top5_accuracy']:>8.3f}"
              f"{summ['macro_f1']:>9.3f}{summ['top1_accuracy_reachable']:>11.3f}"
              f"{summ['n_reachable_classes']:>10}")

    pd.DataFrame(summary_rows).set_index("category").to_csv(os.path.join(out_dir, "metrics_summary.csv"))
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    json.dump(all_json, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print(f"\noutputs -> {out_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="latent space classification validation metrics")
    p.add_argument("--model_root",     required=True, help="Ex: run_v72")
    p.add_argument("--ckpt",           required=True, help="Ex: 2500. Numeric model checkpoint to use within model_root/model/")
    p.add_argument("--dataset_split",  choices=["train", "val", "test"], default="train")
    p.add_argument("--encoded_latents",     action="store_true",
                   help="Use precomputed encoded latents for query (val/test vs train gallery).")
    p.add_argument("--species_list",   default="lizard_species_list.csv")
    p.add_argument("--eval_level", choices=["loo", "specimen", "species", "genus"], 
                   default="specimen", help="Leave one out masking level.")
    p.add_argument("--cm_max_classes", type=int, default=40)
    p.add_argument("--suffix", default=None)
    p.add_argument("--gallery_latents", default=None, help="Path to train (gallery) latents .pth")
    p.add_argument("--query_latents", default=None, help="Path to optimized val/test (query) latents .pth")
    
    args = p.parse_args()
    if args.gallery_latents is None:
        args.gallery_latents = os.path.join(args.model_root, "latent_codes", f"{args.ckpt}.pth")
    if args.query_latents is None:
        if args.encoded_latents:
            args.query_latents = os.path.join(args.model_root, "classification/evaluation/encoded_latents",
                                              f"latent_codes_{args.dataset_split}.pth")
    return args

if __name__ == "__main__":
    main()