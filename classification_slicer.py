import argparse
import json
import os
import sys
import numpy as np
import random

import torch
import pymskt.mesh.meshes as meshes
from NSM.datasets import SDFSamples
from NSM.helper_funcs import fixed_point_coords, load_config, load_model_and_latents, safe_load_mesh_scalars, convert_ply_to_vtk
from NSM.optimization import get_top_k_pcs, optimize_latent, build_sdf_dataset

import pymskt.mesh.meshTools as meshTools
import pyvista as pv

def patch_signed_distance_dtype():
    original = meshTools.pcu.signed_distance_to_mesh
    def signed_distance_to_mesh_patch(pts, points, faces):
        pts = np.asarray(pts, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        return original(pts, points, faces)
    meshTools.pcu.signed_distance_to_mesh = signed_distance_to_mesh_patch

MESH_EXTENSIONS = (".vtk", ".vtp", ".ply")

def _list_meshes(input_dir):
    return [os.path.join(input_dir, name)
            for name in sorted(os.listdir(input_dir))
            if name.lower().endswith(MESH_EXTENSIONS)]

def _load_model_bundle(args, device):
    config = load_config(args.config)
    print("Classification model: {}".format(args.model))
    print("Classification latent codes: {}".format(args.latent_codes))
    print("Classification device: {}".format(device))
    print("Classification iterations: {}".format(args.iterations))
    model, _, latent_codes = load_model_and_latents(args.model, args.latent_codes, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
    return config, model, latent_codes, mean_latent, top_k_reg

def _classify_one(mesh_path, config, model, latent_codes, mean_latent, top_k_reg, device, args, latent_path=None):
    if latent_path and os.path.isfile(latent_path):
        print("Loading precomputed latent from: {}".format(latent_path))
        latent = torch.as_tensor(np.load(latent_path), dtype=torch.float32, device=device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
    else:
        _, mesh_path = convert_ply_to_vtk(mesh_path)
        print("Classification mesh: {}".format(mesh_path))
        points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(mesh_path, config, config["n_pts_per_object"])
        latent = optimize_latent(model, points, sdf_vals, config["latent_size"], 
                                 top_k_reg, mean_latent, latent_codes, iters=args.iterations,
                                 lr=args.learning_rate, device=device)

    codes = latent_codes.to(device)
    distances = 1.0 - torch.nn.functional.cosine_similarity(codes, latent.to(device), dim=1)
    count = min(5, len(distances))
    indices = torch.argsort(distances)[:count].detach().cpu().tolist()
    training_paths = config.get("list_mesh_paths", [])
    matches = []
    for rank, index in enumerate(indices, start=1):
        training_path = training_paths[index] if index < len(training_paths) else ""
        matches.append({
            "rank": rank,
            "latent_index": index,
            "mesh_name": os.path.basename(training_path) or "Unknown",
            "training_path": training_path,
            "cosine_distance": float(distances[index].detach().cpu()),
        })
    return matches, latent.detach().cpu().numpy(), indices

def _write_result_json(path, matches):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"metric": "cosine distance", "matches": matches}, stream, indent=2)

def classify(args):
    repository_root = os.path.dirname(os.path.abspath(__file__))
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)
    patch_signed_distance_dtype()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, model, latent_codes, mean_latent, top_k_reg = _load_model_bundle(args, device)
    all_latents_path = os.path.join(args.output_dir, "all_latents.npy")
    np.save(all_latents_path, latent_codes.detach().cpu().numpy())

    if args.input_dir:
        mesh_files = _list_meshes(args.input_dir)
        if not mesh_files:
            raise ValueError("No mesh files found in folder: {}".format(args.input_dir))
        print("Bulk classification of {} meshes from {}".format(len(mesh_files), args.input_dir))
        results = []
        for i, mesh_file in enumerate(mesh_files, start=1):
            base = os.path.splitext(os.path.basename(mesh_file))[0]
            print("\n[{}/{}] Classifying {}".format(i, len(mesh_files), base))
            try:
                matches, latent_np, indices = _classify_one(
                    mesh_file, config, model, latent_codes, mean_latent, top_k_reg, device, args)
            except Exception as error:  # keep going so one bad mesh doesn't abort the batch
                print("  Failed to classify {}: {}".format(base, error))
                continue
            fossil_path = os.path.join(args.output_dir, base + "_fossil_latent.npy")
            indices_path = os.path.join(args.output_dir, base + "_top5_indices.npy")
            result_json = os.path.join(args.output_dir, base + "_top5.json")
            np.save(fossil_path, latent_np)
            np.save(indices_path, np.array(indices))
            _write_result_json(result_json, matches)
            results.append({
                "input_name": os.path.basename(mesh_file),
                "input_path": os.path.abspath(mesh_file),
                "result_json": result_json,
                "fossil_latent": fossil_path,
                "top5_indices": indices_path,
                "matches": matches,
            })
        with open(args.result, "w", encoding="utf-8") as stream:
            json.dump({
                "metric": "cosine distance",
                "all_latents": all_latents_path,
                "results": results,
            }, stream, indent=2)
        print("\nBulk classification results written to " + args.result)
    else:
        matches, latent_np, indices = _classify_one(
            args.input_mesh, config, model, latent_codes, mean_latent, top_k_reg, device, args,
            latent_path=args.fossil_latent)
        np.save(os.path.join(args.output_dir, "fossil_latent.npy"), latent_np)
        np.save(os.path.join(args.output_dir, "top5_indices.npy"), np.array(indices))
        _write_result_json(args.result, matches)
        print("Classification results written to " + args.result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--latent_codes", required=True)
    parser.add_argument("--input_mesh")
    parser.add_argument("--input_dir")
    parser.add_argument("--fossil_latent",
                        help="Path to a precomputed latent .npy (e.g. from Shape Completion). "
                             "When set with --input_mesh, skips SDF sampling + latent optimization.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    if not args.input_mesh and not args.input_dir:
        parser.error("Provide --input_mesh (single file) or --input_dir (folder).")
    classify(args)