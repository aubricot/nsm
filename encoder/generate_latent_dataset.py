"""Build the encoder-training set (surface points -> latent) from a trained decoder."""

import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NSM.helper_funcs import load_config, load_model_and_latents
from NSM.mesh import create_mesh


def surface_points_from_latent(model, latent, n_pts_per_axis, device):
    """Marching-cubes the decoder at `latent` and return surface vertices (N,3)."""
    with torch.no_grad():
        mesh = create_mesh(
            decoder=model,
            latent_vector=latent,
            n_pts_per_axis=n_pts_per_axis,
            scale_to_original_mesh=False,
            device=device,
            verbose=False,
        )
    if mesh is None:
        return None
    pts = np.asarray(mesh.point_coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 32:
        return None
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run_v44/model_params_config.json")
    ap.add_argument("--model", default="run_v44/model/3000.pth")
    ap.add_argument("--latent_codes", default="run_v44/latent_codes/3000.pth")
    ap.add_argument("--out", default="encoder/data/latent_surface_dataset.pt")
    ap.add_argument("--n_pts_per_axis", type=int, default=128,
                    help="Marching cubes resolution.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N latents.")
    ap.add_argument("--checkpoint_every", type=int, default=25,
                    help="Checkpoint to <out>.partial every N latents.")
    ap.add_argument("--restart", action="store_true",
                    help="Ignore existing <out>.partial and restart.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    model, _, latent_codes = load_model_and_latents(
        args.model, args.latent_codes, config, device
    )
    model.eval()

    n = latent_codes.shape[0]
    if args.limit is not None:
        n = min(n, args.limit)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    partial_path = args.out + ".partial"

    meta = {
        "n_pts_per_axis": args.n_pts_per_axis,
        "latent_size": int(latent_codes.shape[1]),
        "source_latents": args.latent_codes,
        "source_model": args.model,
    }

    # resume from a compatible partial checkpoint if present
    points_list, kept_latents, kept_idx = [], [], []
    start = 0
    if os.path.exists(partial_path) and not args.restart:
        ck = torch.load(partial_path, weights_only=False)
        if ck["meta"]["n_pts_per_axis"] != args.n_pts_per_axis:
            print(f"Ignoring {partial_path}: res {ck['meta']['n_pts_per_axis']} "
                  f"!= requested {args.n_pts_per_axis}")
        else:
            points_list = [p.numpy() for p in ck["points"]]
            kept_latents = [z.numpy() for z in ck["latents"]]
            kept_idx = list(ck["meta"]["kept_idx"])
            start = ck["meta"]["next_idx"]
            print(f"Resuming from {partial_path}: {len(points_list)} shapes kept, "
                  f"restarting at latent {start}")

    print(f"Generating surfaces for latents [{start}, {n}) at "
          f"res={args.n_pts_per_axis} on {device}")

    def flush(next_idx, final=False):
        payload = {
            "latents": torch.from_numpy(np.stack(kept_latents)).float()
                       if kept_latents else torch.empty(0, meta["latent_size"]),
            "points": [torch.from_numpy(p).float() for p in points_list],
            "meta": {**meta, "kept_idx": kept_idx, "next_idx": next_idx},
        }
        path = args.out if final else partial_path
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic write

    t0 = time.time()
    for i in range(start, n):
        z = latent_codes[i : i + 1].to(device)
        pts = surface_points_from_latent(model, z, args.n_pts_per_axis, device)
        if pts is not None:
            points_list.append(pts)
            kept_latents.append(latent_codes[i].cpu().numpy())
            kept_idx.append(i)
        else:
            print(f"  [{i}] skipped (no valid surface)")
        done = i - start + 1
        if done % args.checkpoint_every == 0 or i == n - 1:
            rate = done / (time.time() - t0)
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] kept={len(points_list)} "
                  f"({rate:.2f} shape/s, eta {eta/60:.1f} min)")
            flush(next_idx=i + 1)

    flush(next_idx=n, final=True)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Saved {len(points_list)} shapes -> {args.out} "
          f"({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
