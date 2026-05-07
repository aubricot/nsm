"""
Unified vertebra classification script.

Supports:
- Baseline distance-based classification
- Hierarchy-aware parsing and optional supervised classifiers
- Categorical or normalized spinal position evaluation

ARGUMENTS:
    --run_name TEXT
        Training run directory containing model checkpoint files, latent codes,
        and model_params_config.json. Default: run_v57a

    --checkpoint TEXT
        Checkpoint epoch to load (example: 3000). If omitted, the script
        automatically picks the highest numeric checkpoint in run_name/model/.

    --mesh_split {train,val,test}
        Which split from model_params_config.json to classify. Default: test

    --suffix TEXT
        Suffix used in output file/folder names. Default: timestamp

    --save_vtk {true,false}
        Whether to reconstruct and save decoded VTK meshes for each classified
        sample. Default: false

    --classification_mode {baseline,hierarchy}
        baseline: uses NSM helper label parsing and distance retrieval only.
        hierarchy: uses hierarchy taxonomy parsing and adds genus fields.
        Default: baseline

    --position_mode {categorical,normalized}
        categorical: uses region/label matching only.
        normalized: computes continuous normalized position error using
        vertebrae_counts.csv via SpinePositionMapper. Default: normalized

    --enable_classifiers
        If set (and classification_mode=hierarchy), trains supervised
        classifiers (KNN, SVM, RF, MLP, LR) on training latents and reports
        their predictions for each novel mesh.

    --init {none,family,genus,species}
        Initialization method for latent optimization.
        none: PCA initialization from mean latent and top PCs.
        family/genus/species: initialize from precomputed group-average latents
        in run_name/latent_codes_average/<level>/mean_latents_by_<level>.npz.
        Falls back to PCA if no match/file is found. Default: none

    --latent_noise_std FLOAT
        Standard deviation of Gaussian noise added to the initial latent code
        before optimization. Default: 0.0

    --sample_noise_std FLOAT
        Standard deviation of Gaussian noise added to sampled xyz points before
        latent optimization. Default: 0.0

EXAMPLES:
    python classify_vertebrae.py --run_name run_v57a
    python classify_vertebrae.py --run_name run_v57a --classification_mode hierarchy --enable_classifiers
    python classify_vertebrae.py --run_name run_v57a --position_mode categorical --checkpoint 3000
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from NSM.datasets import SDFSamples
from NSM.helper_funcs import (
    convert_ply_to_vtk,
    load_config,
    load_model_and_latents,
    parse_labels_from_filepaths,
)
from NSM.mesh import create_mesh, get_sdfs
from NSM.optimization import get_top_k_pcs, pca_initialize_latent

from hierarchy.classifiers import predict_classifiers, train_classifiers
from hierarchy.spine_position import SpinePositionMapper
from hierarchy.taxonomy import parse_taxonomy_from_filename

def _latest_checkpoint(run_dir: str) -> str:
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model directory: {model_dir}")
    ckpts = []
    for fname in os.listdir(model_dir):
        stem, ext = os.path.splitext(fname)
        if ext == ".pth" and stem.isdigit():
            ckpts.append(int(stem))
    if not ckpts:
        raise FileNotFoundError(f"No numeric checkpoints found in: {model_dir}")
    return str(max(ckpts))


def _parse_args():
    parser = argparse.ArgumentParser(description="Unified NSM vertebrae classification")
    parser.add_argument("--run_name", type=str, default="run_v57a")
    parser.add_argument("--checkpoint", type=str, default=None, help="If omitted, uses latest.")
    parser.add_argument("--mesh_split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--suffix", type=str, default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    parser.add_argument("--save_vtk", type=lambda x: x.lower() != "false", default=False)

    parser.add_argument("--classification_mode", choices=["baseline", "hierarchy"], default="baseline")
    parser.add_argument("--position_mode", choices=["categorical", "normalized"], default="normalized")
    parser.add_argument("--enable_classifiers", action="store_true")

    parser.add_argument("--init", choices=["none", "family", "genus", "species"], default="none")
    parser.add_argument("--latent_noise_std", type=float, default=0.0)
    parser.add_argument("--sample_noise_std", type=float, default=0.0)
    return parser.parse_args()


def _load_taxonomic_latents(run_dir: str, level: str) -> Optional[Dict[str, np.ndarray]]:
    if level == "none":
        return None
    npz_path = os.path.join(run_dir, "latent_codes_average", level, f"mean_latents_by_{level}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return {name: latent for name, latent in zip(data[level], data["latents"])}


def _init_latent(
    mesh_path: str,
    init_method: str,
    taxonomic_latents: Optional[Dict[str, np.ndarray]],
    mean_latent: torch.Tensor,
    latent_codes: torch.Tensor,
    top_k_reg: int,
    device: str,
) -> Tuple[torch.Tensor, str]:
    if init_method == "none":
        return pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg), "pca_mean"

    parsed = parse_taxonomy_from_filename(os.path.basename(mesh_path))
    if parsed is None or taxonomic_latents is None:
        return pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg), "pca_fallback"

    key = parsed.get(init_method)
    if key in taxonomic_latents:
        return torch.from_numpy(taxonomic_latents[key]).float().unsqueeze(0).to(device), key
    return pca_initialize_latent(mean_latent, latent_codes, top_k=top_k_reg), "pca_fallback"


def _optimize_latent(
    decoder,
    points: torch.Tensor,
    sdf_vals: torch.Tensor,
    init_latent: torch.Tensor,
    latent_noise_std: float,
    device: str,
    iters: int = 1000,
    lr: float = 1e-3,
):
    latent = init_latent.clone().detach()
    if latent_noise_std > 0:
        latent = latent + torch.randn_like(latent) * latent_noise_std
    latent = latent.to(device).requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr)

    points = points.to(device)
    sdf_vals = sdf_vals.to(device)
    decoder = decoder.to(device)

    for _ in range(iters):
        optimizer.zero_grad()
        inputs = torch.cat([latent.expand(points.shape[0], -1), points], dim=1)
        pred_sdf = decoder(inputs)
        loss = F.l1_loss(pred_sdf.squeeze(), sdf_vals.squeeze())
        loss.backward()
        optimizer.step()
    return latent.detach()


def _build_inference_dataset(config: dict, mesh_path: str):
    return SDFSamples(
        list_mesh_paths=[mesh_path],
        multiprocessing=False,
        subsample=config["samples_per_object_per_batch"],
        print_filename=True,
        n_pts=config["n_pts_per_object"],
        p_near_surface=config["percent_near_surface"],
        p_further_from_surface=config["percent_further_from_surface"],
        sigma_near=config["sigma_near"],
        sigma_far=config["sigma_far"],
        rand_function=config["random_function"],
        center_pts=config["center_pts"],
        norm_pts=config["normalize_pts"],
        reference_mesh=None,
        verbose=config["verbose"],
        save_cache=config["cache"],
        equal_pos_neg=config["equal_pos_neg"],
        fix_mesh=config["fix_mesh"],
    )


def _extract_hierarchy_labels(fname: str):
    parsed = parse_taxonomy_from_filename(fname)
    if not parsed:
        return None, None, None
    return parsed.get("species"), parsed.get("genus"), parsed.get("position")


def main():
    args = _parse_args()
    run_dir = args.run_name
    ckpt = args.checkpoint if args.checkpoint else _latest_checkpoint(run_dir)

    config = load_config(config_path=f"{run_dir}/model_params_config.json")
    device = config.get("device", "cuda:0")
    model_path = f"{run_dir}/model/{ckpt}.pth"
    lc_path = f"{run_dir}/latent_codes/{ckpt}.pth"

    model, _, latent_codes = load_model_and_latents(model_path, lc_path, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

    mesh_paths = config[f"{args.mesh_split}_paths"] if args.mesh_split != "train" else config["list_mesh_paths"]
    train_paths = config["list_mesh_paths"]
    train_files = [os.path.basename(p) for p in train_paths]

    position_mapper = None
    if args.position_mode == "normalized":
        counts_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertebrae_counts.csv")
        if os.path.exists(counts_csv):
            position_mapper = SpinePositionMapper(counts_csv)

    taxonomic_latents = _load_taxonomic_latents(run_dir, args.init)

    clf_models = None
    if args.enable_classifiers and args.classification_mode == "hierarchy":
        y_rows = []
        x_rows = []
        latents_np = latent_codes.detach().cpu().numpy()
        for idx, f in enumerate(train_files):
            species, genus, position = _extract_hierarchy_labels(f)
            if species and genus and position:
                y_rows.append([species, genus, position])
                x_rows.append(latents_np[idx])
        if len(y_rows) >= 2:
            clf_models, _ = train_classifiers(np.array(x_rows), np.array(y_rows))

    results_dir = f"{run_dir}/classify_vertebrae/results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"summary_{args.classification_mode}_{args.position_mode}_{args.suffix}.csv")

    summary = []
    train_tensor = latent_codes.detach().to(device).float()
    for mesh_path in mesh_paths:
        if mesh_path.endswith(".ply"):
            _, mesh_path = convert_ply_to_vtk(mesh_path)

        init_latent, init_source = _init_latent(
            mesh_path, args.init, taxonomic_latents, mean_latent, latent_codes, top_k_reg, device
        )

        sdf_dataset = _build_inference_dataset(config, mesh_path)
        sample, _ = sdf_dataset[0]
        points = sample["xyz"].float().to(device)
        if args.sample_noise_std > 0:
            points = points + torch.randn_like(points) * args.sample_noise_std
        sdf_vals = sample["gt_sdf"]

        latent_novel = _optimize_latent(
            model, points, sdf_vals, init_latent, args.latent_noise_std, device
        )

        with torch.no_grad():
            novel_tensor = latent_novel.float().to(device)
            sims = F.cosine_similarity(novel_tensor, train_tensor)
            top5_idx = torch.argsort(sims, descending=True)[:5].cpu().numpy()
            top5_dist = (1 - sims[top5_idx]).cpu().numpy()

        top_names = [train_files[i] for i in top5_idx]
        mesh_name = os.path.basename(mesh_path)

        if args.classification_mode == "hierarchy":
            gt_species, gt_genus, gt_pos = _extract_hierarchy_labels(mesh_name)
            pred_species, pred_genus, pred_pos = _extract_hierarchy_labels(top_names[0])
        else:
            gt_pair, _ = parse_labels_from_filepaths([mesh_name])
            pred_pair, _ = parse_labels_from_filepaths([top_names[0]])
            gt_species, gt_pos = gt_pair[0]
            pred_species, pred_pos = pred_pair[0]
            gt_genus = None
            pred_genus = None

        row = {
            "mesh": mesh_name,
            "init_source": init_source,
            "top1_name": top_names[0],
            "top1_distance": float(top5_dist[0]),
            "top1_species_match": "yes" if gt_species and gt_species == pred_species else "no",
            "top1_position_match": "yes" if gt_pos and pred_pos and gt_pos == pred_pos else "no",
            "gt_species": gt_species,
            "pred_species": pred_species,
            "gt_position": gt_pos,
            "pred_position": pred_pos,
        }
        if args.classification_mode == "hierarchy":
            row["top1_genus_match"] = "yes" if gt_genus and gt_genus == pred_genus else "no"
            row["gt_genus"] = gt_genus
            row["pred_genus"] = pred_genus

        if args.position_mode == "normalized" and position_mapper is not None:
            gt_norm = position_mapper.get_normalized_position(mesh_name)
            pred_norm = position_mapper.get_normalized_position(top_names[0])
            row["gt_norm_position"] = gt_norm
            row["pred_norm_position"] = pred_norm
            row["top1_norm_abs_error"] = (
                abs(gt_norm - pred_norm) if gt_norm is not None and pred_norm is not None else None
            )

        for rank, (name, dist) in enumerate(zip(top_names, top5_dist), start=1):
            row[f"top{rank}_name"] = name
            row[f"top{rank}_distance"] = float(dist)

        if clf_models is not None:
            preds, _, _ = predict_classifiers(clf_models, latent_novel.detach().cpu().numpy())
            for clf_name, clf_pred in preds.items():
                row[f"{clf_name}_pred_species"] = clf_pred[0][0]
                row[f"{clf_name}_pred_genus"] = clf_pred[0][1]
                row[f"{clf_name}_pred_position"] = clf_pred[0][2]

        summary.append(row)

        if args.save_vtk:
            out_dir = f"{run_dir}/classify_vertebrae/predictions_{args.suffix}/{os.path.splitext(mesh_name)[0]}"
            os.makedirs(out_dir, exist_ok=True)
            mesh_out = create_mesh(
                decoder=model,
                latent_vector=latent_novel,
                n_pts_per_axis=256,
                voxel_origin=(-1.0, -1.0, -1.0),
                voxel_size=2.0 / 255,
                path_original_mesh=None,
                offset=np.array([0.0, 0.0, 0.0]),
                scale=1.0,
                icp_transform=None,
                objects=1,
                verbose=False,
                device=device,
            )
            if isinstance(mesh_out, list):
                mesh_out = mesh_out[0]
            mesh_out.save(os.path.join(out_dir, f"{os.path.splitext(mesh_name)[0]}_decoded.vtk"))

    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
