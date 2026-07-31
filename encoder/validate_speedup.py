"""Head-to-head: one-shot encoder vs. test-time latent optimization (chamfer + time)."""

import os
import sys
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NSM.helper_funcs import load_config, load_model_and_latents
from NSM.mesh import create_mesh
from NSM.optimization import optimize_latent_partial
from encoder.pointnet_encoder import PointNetEncoder
from encoder.train_encoder import random_crop, resample


def surface(model, z, res, device):
    with torch.no_grad():
        m = create_mesh(decoder=model, latent_vector=z, n_pts_per_axis=res,
                        scale_to_original_mesh=False, device=device, verbose=False)
    return np.asarray(m.point_coords, dtype=np.float32) if m is not None else None


def chamfer(a, b, n=2000):
    a = torch.from_numpy(a[np.random.choice(len(a), min(n, len(a)), replace=False)])
    b = torch.from_numpy(b[np.random.choice(len(b), min(n, len(b)), replace=False)])
    d = torch.cdist(a, b)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2


def sdf_for_points(pts):
    """Surface points have SDF 0; that's the target the optimizer fits."""
    return torch.zeros(pts.shape[0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="encoder/data/subset.pt")
    ap.add_argument("--encoder", default="encoder/checkpoints/subset_enc.pt")
    ap.add_argument("--config", default="run_v44/model_params_config.json")
    ap.add_argument("--model", default="run_v44/model/3000.pth")
    ap.add_argument("--latent_codes", default="run_v44/latent_codes/3000.pth")
    ap.add_argument("--n_shapes", type=int, default=5)
    ap.add_argument("--opt_iters", type=int, default=3000,
                    help="Optimizer iters to compare against (full pipeline uses 11000).")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    model, _, latent_codes = load_model_and_latents(
        args.model, args.latent_codes, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = latent_codes.std().mean()

    payload = torch.load(args.data)
    points, latents = payload["points"], payload["latents"]
    ckpt = torch.load(args.encoder, map_location=device)
    enc = PointNetEncoder(latent_size=ckpt["latent_size"]).to(device)
    enc.load_state_dict(ckpt["model"]); enc.eval()
    n_points = ckpt["n_points"]

    idxs = np.random.choice(len(points), args.n_shapes, replace=False)
    print(f"{'shape':>6} | {'enc_cham':>9} {'enc_t(s)':>8} | {'opt_cham':>9} {'opt_t(s)':>8}")
    e_c, e_t, o_c, o_t = [], [], [], []
    for si in idxs:
        z_true = latents[si:si+1].to(device)
        gt = surface(model, z_true, args.res, device)

        partial = random_crop(points[si])            # partial/broken input
        partial = resample(partial, n_points)

        # encoder: one forward pass
        t = time.time()
        with torch.no_grad():
            z_enc = enc(partial.unsqueeze(0).to(device))
        te = time.time() - t
        rec_e = surface(model, z_enc, args.res, device)
        ce = chamfer(gt, rec_e) if rec_e is not None else float("nan")

        # optimizer: shape-completion path
        pts_dev = partial.to(device)
        sdf = sdf_for_points(partial).to(device)
        t = time.time()
        z_opt, _ = optimize_latent_partial(
            model, pts_dev, sdf, ckpt["latent_size"], mean_latent=mean_latent,
            latent_init=latent_codes, top_k=200, iters=args.opt_iters, lr=1e-4,
            lambda_reg=1e-3, clamp_val=1.0, latent_std=latent_std,
            scheduler_step=800, scheduler_gamma=0.9, batch_inference_size=32768,
            multi_stage=False, verbose=False, device=device)
        to = time.time() - t
        rec_o = surface(model, z_opt, args.res, device)
        co = chamfer(gt, rec_o) if rec_o is not None else float("nan")

        e_c.append(ce); e_t.append(te); o_c.append(co); o_t.append(to)
        print(f"{si:>6} | {ce:>9.4f} {te:>8.3f} | {co:>9.4f} {to:>8.2f}")

    print("-" * 52)
    print(f"{'mean':>6} | {np.mean(e_c):>9.4f} {np.mean(e_t):>8.3f} | "
          f"{np.mean(o_c):>9.4f} {np.mean(o_t):>8.2f}")
    print(f"\nSpeedup (encoder vs {args.opt_iters}-iter opt): "
          f"{np.mean(o_t)/max(np.mean(e_t),1e-6):.0f}x faster")
    print(f"Note: production pipeline runs 11000 iters, so real speedup is "
          f"~{11000/args.opt_iters * np.mean(o_t)/max(np.mean(e_t),1e-6):.0f}x")


if __name__ == "__main__":
    main()
