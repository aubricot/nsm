"""Train the encoder to predict DeepSDF latents from (randomly cropped) surface points."""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder.pointnet_encoder import PointNetEncoder


def random_crop(pts, min_keep=0.4, max_keep=0.85):
    """Simulate a partial scan by slicing off a slab along a random direction."""
    n = pts.shape[0]
    d = torch.randn(3)
    d = d / (d.norm() + 1e-8)
    proj = pts @ d
    keep_frac = float(np.random.uniform(min_keep, max_keep))
    k = max(int(n * keep_frac), 32)
    idx = torch.topk(proj, k, largest=bool(np.random.rand() < 0.5)).indices
    return pts[idx]


def resample(pts, n_points):
    """Sample exactly n_points (with replacement if the cloud is smaller)."""
    n = pts.shape[0]
    replace = n < n_points
    idx = np.random.choice(n, n_points, replace=replace)
    return pts[idx]


class SurfaceLatentDataset(Dataset):
    def __init__(self, points, latents, n_points=2048, augment=True,
                 p_crop=0.6, noise_std=0.005):
        self.points = points
        self.latents = latents
        self.n_points = n_points
        self.augment = augment
        self.p_crop = p_crop
        self.noise_std = noise_std

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        pts = self.points[i]
        if self.augment and np.random.rand() < self.p_crop:
            pts = random_crop(pts)
        pts = resample(pts, self.n_points)
        if self.augment and self.noise_std > 0:
            pts = pts + torch.randn_like(pts) * self.noise_std
        return pts, self.latents[i]


def latent_loss(pred, target):
    mse = F.mse_loss(pred, target)
    cos = (1 - F.cosine_similarity(pred, target, dim=1)).mean()
    return mse + 0.1 * cos, mse, cos


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, tot_mse, tot_cos, nb = 0.0, 0.0, 0.0, 0
    for pts, tgt in loader:
        pts, tgt = pts.to(device), tgt.to(device)
        with torch.set_grad_enabled(train):
            pred = model(pts)
            loss, mse, cos = latent_loss(pred, tgt)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot += loss.item(); tot_mse += mse.item(); tot_cos += cos.item(); nb += 1
    return tot / nb, tot_mse / nb, tot_cos / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="encoder/data/latent_surface_dataset.pt")
    ap.add_argument("--out", default="encoder/checkpoints/encoder.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    payload = torch.load(args.data)
    points, latents = payload["points"], payload["latents"]
    latent_size = latents.shape[1]
    n = len(points)
    print(f"Loaded {n} shapes, latent_size={latent_size}")

    perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = SurfaceLatentDataset([points[i] for i in tr_idx], latents[tr_idx],
                              n_points=args.n_points, augment=True)
    va = SurfaceLatentDataset([points[i] for i in val_idx], latents[val_idx],
                              n_points=args.n_points, augment=False)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    model = PointNetEncoder(latent_size=latent_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        tl, tmse, tcos = run_epoch(model, tr_loader, device, opt)
        vl, vmse, vcos = run_epoch(model, va_loader, device, None)
        sched.step()
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(),
                        "latent_size": latent_size,
                        "n_points": args.n_points}, args.out)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | train {tl:.4f} (mse {tmse:.4f} cos {tcos:.4f}) "
                  f"| val {vl:.4f} (mse {vmse:.4f} cos {vcos:.4f}) | best {best:.4f} "
                  f"| {time.time()-t0:.0f}s")
    print(f"Done. Best val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
