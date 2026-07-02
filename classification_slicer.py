import argparse
import json
import os
import sys


def set_reproducible_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def patch_signed_distance_dtype():
    import numpy as np
    import pymskt.mesh.meshTools as meshTools

    original = meshTools.pcu.signed_distance_to_mesh

    def signed_distance_to_mesh_patch(pts, points, faces):
        pts = np.asarray(pts, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        return original(pts, points, faces)

    meshTools.pcu.signed_distance_to_mesh = signed_distance_to_mesh_patch


def _prepare_mesh(mesh_path, output_dir):
    """Return a VTK mesh path; conversion products are kept with the results."""
    if not mesh_path.lower().endswith(".ply"):
        return mesh_path
    import pyvista as pv

    converted = os.path.join(
        output_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".vtk"
    )
    pv.read(mesh_path).save(converted)
    return converted


def classify(args):
    repository_root = os.path.dirname(os.path.abspath(__file__))
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

    import torch
    import pymskt.mesh.meshes as meshes
    from NSM.datasets import SDFSamples
    from NSM.helper_funcs import (
        fixed_point_coords,
        load_config,
        load_model_and_latents,
        safe_load_mesh_scalars,
    )
    from NSM.optimization import get_top_k_pcs, optimize_latent

    meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
    meshes.Mesh.point_coords = property(fixed_point_coords)
    patch_signed_distance_dtype()

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_path = _prepare_mesh(args.input_mesh, args.output_dir)
    config = load_config(args.config)
    set_reproducible_seed(args.seed)
    print("Classification seed: {}".format(args.seed))
    print("Classification mesh: {}".format(mesh_path))
    print("Classification model: {}".format(args.model))
    print("Classification latent codes: {}".format(args.latent_codes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Classification device: {}".format(device))
    print("Classification iterations: {}".format(args.iterations))
    model, _, latent_codes = load_model_and_latents(args.model, args.latent_codes, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

    dataset = SDFSamples(
        list_mesh_paths=[mesh_path], multiprocessing=False,
        subsample=config["samples_per_object_per_batch"], print_filename=True,
        n_pts=config["n_pts_per_object"], p_near_surface=config["percent_near_surface"],
        p_further_from_surface=config["percent_further_from_surface"],
        sigma_near=config["sigma_near"], sigma_far=config["sigma_far"],
        rand_function=config["random_function"], center_pts=config["center_pts"],
        norm_pts=config["normalize_pts"], scale_method=config["scale_method"],
        reference_mesh=None, verbose=config["verbose"], save_cache=config["cache"],
        equal_pos_neg=config["equal_pos_neg"], fix_mesh=config["fix_mesh"],
        random_seed=args.seed,
    )
    set_reproducible_seed(args.seed)
    sample, _ = dataset[0]
    set_reproducible_seed(args.seed)
    latent = optimize_latent(
        model, sample["xyz"].to(device), sample["gt_sdf"].to(device),
        config["latent_size"], top_k_reg, mean_latent, latent_codes,
        iters=args.iterations, lr=args.learning_rate, device=device,
    )

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

    with open(args.result, "w", encoding="utf-8") as stream:
        json.dump({"metric": "cosine distance", "matches": matches}, stream, indent=2)
    print("Classification results written to " + args.result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--latent_codes", required=True)
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    classify(parser.parse_args())
