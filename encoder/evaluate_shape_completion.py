"""Shape-completion eval: one-shot encoder vs. full 2-phase optimizer, scored by
chamfer-to-GT and time. Outputs PNGs, .vtk meshes, results.csv, summary.md."""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NSM.helper_funcs import load_config, load_model_and_latents
from NSM.mesh import create_mesh
from NSM.datasets import SDFSamples
from NSM.optimization import get_top_k_pcs, optimize_latent_partial
from encoder.pointnet_encoder import PointNetEncoder


def chamfer(a, b, n=3000):
    a = torch.from_numpy(a[np.random.choice(len(a), min(n, len(a)), replace=False)])
    b = torch.from_numpy(b[np.random.choice(len(b), min(n, len(b)), replace=False)])
    d = torch.cdist(a, b)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2


def recon_surface(model, z, res, device):
    m = create_mesh(decoder=model, latent_vector=z, n_pts_per_axis=res,
                    scale_to_original_mesh=False, device=device, verbose=False)
    if m is None:
        return None, None
    return np.asarray(m.point_coords, dtype=np.float32), m


def encode(enc, n_points, points, sdf, device, surf_thresh=0.01):
    """Encoder fast-mode latent: near-surface points -> one forward pass."""
    abs_sdf = sdf.abs().squeeze()
    surf = points[abs_sdf < surf_thresh]
    if surf.shape[0] < 32:
        k = min(n_points, points.shape[0])
        surf = points[torch.topk(abs_sdf, k, largest=False).indices]
    sel = np.random.choice(surf.shape[0], n_points, replace=surf.shape[0] < n_points)
    with torch.no_grad():
        return enc(surf[sel].unsqueeze(0).to(device))


def crop_partial(points, sdf, keep=0.65, rng=None):
    """Drop the samples on one side of a random plane -> a partial 'broken' shape."""
    rng = rng or np.random
    surf_mask = sdf.abs().squeeze().cpu().numpy() < 0.01
    center = points[surf_mask].mean(0) if surf_mask.any() else points.mean(0)
    normal = torch.tensor(rng.randn(3), dtype=torch.float32)
    normal = normal / normal.norm()
    proj = (points.cpu() - center.cpu()) @ normal
    thresh = torch.quantile(proj, 1.0 - keep)
    mask = proj > thresh
    return points[mask], sdf[mask]


def render(out_png, gt_path, part_pts, enc_path, opt_path):
    """2x2 panel: GT / partial input points / encoder & optimizer completions."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(900, 900))
    panels = [("Ground truth", gt_path, "lightgray"),
              ("Partial input", part_pts, "red"),
              ("Encoder (fast)", enc_path, "deepskyblue"),
              ("Optimizer (full)", opt_path, "limegreen")]
    for idx, (title, obj, color) in enumerate(panels):
        pl.subplot(idx // 2, idx % 2)
        pl.add_text(title, font_size=10)
        if isinstance(obj, str):
            if obj and os.path.exists(obj):
                pl.add_mesh(pv.read(obj), color=color, smooth_shading=True)
        elif obj is not None and len(obj):
            pl.add_points(pv.PolyData(np.asarray(obj, dtype=np.float32)),
                          color=color, point_size=3, render_points_as_spheres=True)
    pl.link_views()
    pl.camera_position = "xy"
    pl.screenshot(out_png)
    pl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run_v44/model_params_config.json")
    ap.add_argument("--model", default="run_v44/model/3000.pth")
    ap.add_argument("--latent_codes", default="run_v44/latent_codes/3000.pth")
    ap.add_argument("--encoder", default="encoder/checkpoints/encoder.pt")
    ap.add_argument("--out_dir", default="encoder/data/shape_completion_eval")
    ap.add_argument("--n_shapes", type=int, default=5)
    ap.add_argument("--keep", type=float, default=0.65, help="Fraction of samples kept when cropping.")
    ap.add_argument("--n_samples", type=int, default=240, help="Optimizer point budget (production default).")
    ap.add_argument("--phase1_iters", type=int, default=3000)
    ap.add_argument("--phase2_iters", type=int, default=8000)
    ap.add_argument("--res", type=int, default=128, help="Marching-cubes resolution (GT + reconstructions).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.config)
    model, _, latent_codes = load_model_and_latents(args.model, args.latent_codes, config, device)
    model.eval()
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = latent_codes.std().mean()
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

    ckpt = torch.load(args.encoder, map_location=device)
    enc = PointNetEncoder(latent_size=ckpt["latent_size"]).to(device)
    enc.load_state_dict(ckpt["model"]); enc.eval()
    n_points = ckpt["n_points"]

    sel = rng.choice(latent_codes.shape[0], args.n_shapes, replace=False)
    rows = []
    for i in sel:
        z_true = latent_codes[int(i):int(i) + 1].to(device)
        gt_pts, gt_mesh = recon_surface(model, z_true, args.res, device)
        if gt_pts is None:
            print(f"[{int(i)}] skipped (no GT surface)")
            continue
        gt_path = os.path.join(args.out_dir, f"shape_{int(i)}_gt.vtk")
        gt_mesh.save_mesh(gt_path)

        # production SDF sampling on the full GT mesh, then crop to a partial input
        ds = SDFSamples(
            list_mesh_paths=[gt_path], multiprocessing=False,
            subsample=config["samples_per_object_per_batch"], print_filename=False,
            n_pts=config["n_pts_per_object"], p_near_surface=config["percent_near_surface"],
            p_further_from_surface=config["percent_further_from_surface"],
            sigma_near=config["sigma_near"], sigma_far=config["sigma_far"],
            rand_function=config["random_function"], center_pts=config["center_pts"],
            norm_pts=config["normalize_pts"], scale_method=config["scale_method"],
            reference_mesh=None, verbose=False, save_cache=config["cache"],
            equal_pos_neg=config["equal_pos_neg"], fix_mesh=config["fix_mesh"])
        sample, _ = ds[0]
        pts_full = sample["xyz"].to(device).squeeze()
        sdf_full = sample["gt_sdf"].to(device).reshape(-1, 1)
        part_pts, part_sdf = crop_partial(pts_full, sdf_full, keep=args.keep, rng=rng)

        # encoder: one forward pass
        t = time.time()
        z_enc = encode(enc, n_points, part_pts, part_sdf, device)
        t_enc = time.time() - t
        enc_pts, enc_mesh = recon_surface(model, z_enc, args.res, device)

        # optimizer: full 2-phase production pipeline on a small point budget
        idx = torch.randperm(part_pts.shape[0])[:args.n_samples]
        opt_pts_in, opt_sdf_in = part_pts[idx], part_sdf[idx]
        t = time.time()
        z_opt, _ = optimize_latent_partial(
            model, opt_pts_in, opt_sdf_in, config["latent_size"], mean_latent=mean_latent,
            latent_init=latent_codes, top_k=top_k_reg, iters=args.phase1_iters, lr=1e-4,
            lambda_reg=1e-3, clamp_val=1.0, latent_std=latent_std, scheduler_step=800,
            scheduler_gamma=0.9, batch_inference_size=32768, multi_stage=False,
            verbose=False, device=device)
        z_opt, _ = optimize_latent_partial(
            model, opt_pts_in, opt_sdf_in, config["latent_size"], latent_init=z_opt,
            top_k=top_k_reg, iters=args.phase2_iters, lr=1e-5, lambda_reg=1e-5,
            clamp_val=None, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7,
            batch_inference_size=32768, multi_stage=True, verbose=False, device=device)
        t_opt = time.time() - t
        opt_pts, opt_mesh = recon_surface(model, z_opt, args.res, device)

        c_enc = chamfer(gt_pts, enc_pts) if enc_pts is not None else float("nan")
        c_opt = chamfer(gt_pts, opt_pts) if opt_pts is not None else float("nan")
        enc_path = os.path.join(args.out_dir, f"shape_{int(i)}_encoder.vtk")
        opt_path = os.path.join(args.out_dir, f"shape_{int(i)}_optimizer.vtk")
        if enc_mesh is not None:
            enc_mesh.save_mesh(enc_path)
        if opt_mesh is not None:
            opt_mesh.save_mesh(opt_path)
        try:
            surf_part = part_pts[part_sdf.abs().squeeze() < 0.01].cpu().numpy()
            render(os.path.join(args.out_dir, f"shape_{int(i)}_compare.png"),
                   gt_path, surf_part, enc_path, opt_path)
        except Exception as e:
            print(f"  render failed for {int(i)}: {e}")

        rows.append(dict(shape=int(i), enc_chamfer=c_enc, opt_chamfer=c_opt,
                         enc_time=t_enc, opt_time=t_opt, speedup=t_opt / max(t_enc, 1e-6)))
        print(f"[{int(i)}] enc_cham {c_enc:.4f} ({t_enc:.3f}s)  "
              f"opt_cham {c_opt:.4f} ({t_opt:.1f}s)  {t_opt / max(t_enc, 1e-6):.0f}x")

    if not rows:
        print("No shapes evaluated.")
        return

    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    mean = lambda k: float(np.nanmean([r[k] for r in rows]))
    md = [
        "# Shape completion: encoder (fast mode) vs. full optimizer",
        "",
        f"Model `{args.model}`, encoder `{args.encoder}`, {len(rows)} held-out shapes, "
        f"partial input keeps {args.keep:.0%} of samples, reconstruction res {args.res}.",
        "",
        "| Method | Chamfer to GT (mean) | Time/shape (mean) |",
        "|---|---|---|",
        f"| Encoder (fast) | {mean('enc_chamfer'):.4f} | {mean('enc_time'):.3f} s |",
        f"| Optimizer (phase1 {args.phase1_iters} + phase2 {args.phase2_iters}) | "
        f"{mean('opt_chamfer'):.4f} | {mean('opt_time'):.1f} s |",
        "",
        f"**Mean speedup: {mean('speedup'):.0f}x faster** "
        f"(lower chamfer = closer to ground truth).",
        "",
        "Per-shape comparison images: `shape_<id>_compare.png`. "
        "Meshes (`_gt/_encoder/_optimizer.vtk`) open directly in Slicer.",
    ]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("-" * 60)
    print(f"encoder mean chamfer {mean('enc_chamfer'):.4f} | "
          f"optimizer mean chamfer {mean('opt_chamfer'):.4f} | "
          f"{mean('speedup'):.0f}x faster")
    print(f"wrote {csv_path} and summary.md to {args.out_dir}")


if __name__ == "__main__":
    main()
