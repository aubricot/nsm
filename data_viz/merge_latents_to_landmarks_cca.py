import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.stats import chi2 as chi2_dist


# --- Hardcoded input: the landmark CSV in the repo-root (parent of the run folder) ---
LANDMARKS_CSV = "../vtk_to_landmarks.csv"
# Subfolder (inside run_folder) for all CCA outputs - no change required
CCA_SUBDIR = "cca_analysis"

# Other FYI:
# Just point your --run_folder to the folder with the latents and the config
# We look for {run_folder}/latent_codes/3000.pth to merge latent with landmarks
# {run_folder}/model_params_config.json to get the list_mesh_paths ordering for latents

"""
Merge per-shape latent vectors (from a trained run) into the landmark CSV.

We take the basename of each entry in `list_mesh_paths` (without the .vtk
extension), match it against the `matched_vtk_in_csv` column of the landmark
CSV, and append the latent dims as new columns.

The merged table is then kept in memory and fed straight into a non-residual
CCA analysis relating the landmark coordinates (LM cols, PCA-reduced) to the raw latent dims. 
All CCA plots and CSVs are written to {run_folder}/cca_analysis/.

Arguments
---------
--run_folder   (str, REQUIRED)
    Run name (e.g. "run_v57aaaa", resolved as ./run_v57aaaa) or a full/relative
    path to the run folder. Used to locate the latent codes, the model config,
    and to write all outputs.
--latent_name  (str, default: "3000.pth")
    Filename inside {run_folder}/latent_codes/ to load the latent weights from.
    Pass another checkpoint (e.g. "2000.pth") to merge a different epoch.
--skip_cca     (flag, default: off)
    If set, only build and write the merged CSV; skip the CCA analysis entirely.
--n_lm         (int, default: 40)
    Number of top PCA components kept for the landmark (LM) block before CCA.
    Clamped to the number of available LM columns (84 here).
--n_lat        (int, default: 40)
    Number of leading raw latent dimensions used as-is (no PCA) before CCA.
    Clamped to the number of available latent columns (512 here).
--reg          (float, default: 1e-3)
    Ridge regularization lambda added during the whitening eigendecomposition;
    larger values stabilize ill-conditioned covariance matrices.
--n_plot       (int, default: 6)
    Number of canonical variate pairs to draw in the scatter-plot grid.
--n_show       (int, default: 8)
    Number of CCA dimensions shown in the weight/cross-variate heatmaps (the
    scree plot shows up to 2x this many bars).

Usage:
    Easiest with defaults: 
    python merge_latents_to_landmarks_cca.py --run_folder run_v57aaaa


    python merge_latents_to_landmarks_cca.py --run_folder /abs/path/to/run --latent_name 2000.pth
    python merge_latents_to_landmarks_cca.py --run_folder run_v57aaaa --skip_cca
    python merge_latents_to_landmarks_cca.py --run_folder run_v57aaaa --n_lm 30 --n_lat 50
"""


def basename_no_ext(path):
    """vertebrae_meshes_v57_renamed/agamidae_..._01-c3.vtk -> agamidae_..._01-c3"""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_folder', required=True,
                        help='Run name (e.g. "run_v57aaaa", resolved as ./run_v57aaaa) '
                             'or a full/relative path to the run folder.')
    parser.add_argument('--latent_name', default='3000.pth',
                        help='File inside {run_folder}/latent_codes/ to load (default: 3000.pth).')
    # --- CCA analysis options (mirrors non_residual_cca_analysis.py) ---
    parser.add_argument('--skip_cca', action='store_true',
                        help='Only build/write the merged CSV; skip the CCA analysis.')
    parser.add_argument('--n_lm', type=int, default=40,
                        help='Top-K PCA components of the LM block (default 40, max = #LM cols).')
    parser.add_argument('--n_lat', type=int, default=40,
                        help='First-K raw latent dims, no PCA (default 40, max = #latent cols).')
    parser.add_argument('--reg', type=float, default=1e-3,
                        help='Ridge regularization lambda for whitening.')
    parser.add_argument('--n_plot', type=int, default=6,
                        help='Number of canonical pairs to scatter-plot.')
    parser.add_argument('--n_show', type=int, default=8,
                        help='Number of CCA dims shown in heatmaps.')
    args = parser.parse_args()

    # ---- Resolve paths ----
    # If --run_folder is a plain name, treat it as ./{name}; otherwise use as given.
    run_folder = args.run_folder
    if os.path.basename(os.path.normpath(run_folder)) == run_folder.strip('/\\'):
        # plain name with no directory component
        run_folder = os.path.join('.', run_folder)
    run_folder = os.path.normpath(run_folder)

    latent_path = os.path.join(run_folder, 'latent_codes', args.latent_name)
    config_path = os.path.join(run_folder, 'model_params_config.json')

    if not os.path.isfile(latent_path):
        raise FileNotFoundError(f'Latent file not found: {latent_path}')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not os.path.isfile(LANDMARKS_CSV):
        raise FileNotFoundError(f'Landmark CSV not found: {LANDMARKS_CSV}')

    print(f'[paths] run_folder  : {run_folder}')
    print(f'[paths] latent_path : {latent_path}')
    print(f'[paths] config_path : {config_path}')
    print(f'[paths] landmarks   : {LANDMARKS_CSV}')

    # ---- Load latents (same access pattern as load_model_and_latents) ----
    latent_ckpt = torch.load(latent_path, map_location='cpu', weights_only=False)
    weight = latent_ckpt['latent_codes']['weight'].detach().cpu().numpy()
    n_objects, latent_dim = weight.shape
    print(f'[latents] weight shape: {weight.shape} (epoch={latent_ckpt.get("epoch")})')

    # ---- Load config + mesh-path ordering ----
    with open(config_path, 'r') as f:
        config = json.load(f)
    list_mesh_paths = config['list_mesh_paths']
    cfg_latent_size = config.get('latent_size')

    # ---- Sanity checks ----
    if len(list_mesh_paths) != n_objects:
        raise ValueError(
            f'Length mismatch: list_mesh_paths has {len(list_mesh_paths)} entries '
            f'but latent weight has {n_objects} rows. These must align 1:1.')
    if cfg_latent_size is not None and cfg_latent_size != latent_dim:
        raise ValueError(
            f'latent_size mismatch: config says {cfg_latent_size}, weight has {latent_dim} dims.')
    print(f'[check] OK: {n_objects} mesh paths align with {n_objects} latent rows, dim={latent_dim}')

    # ---- Build basename -> latent vector map ----
    basenames = [basename_no_ext(p) for p in list_mesh_paths]
    dup = pd.Series(basenames)
    n_dup = int(dup.duplicated().sum())
    if n_dup:
        print(f'[warn] {n_dup} duplicate basenames in list_mesh_paths; '
              f'the last occurrence wins in the lookup.')
    latent_by_name = {name: weight[i] for i, name in enumerate(basenames)}

    # ---- Load landmark CSV ----
    df = pd.read_csv(LANDMARKS_CSV)
    match_col = 'vtk_basename'
    if match_col not in df.columns:
        raise ValueError(f"CSV has no '{match_col}' column.")
    print(f'[csv] {len(df)} rows, {len(df.columns)} columns loaded')
    print(f'[csv] matching latents on column: {match_col}')

    # ---- latent_found flag ----
    latent_found = df[match_col].isin(latent_by_name.keys())

    # Insert latent_found just before 'landmark_found' if present, else before
    # the match column 'matched_vtk_in_csv', else at the front.
    if 'landmark_found' in df.columns:
        insert_at = df.columns.get_loc('landmark_found')
    elif match_col in df.columns:
        insert_at = df.columns.get_loc(match_col)
    else:
        print("[warn] no 'landmark_found'/match column; inserting 'latent_found' at the front.")
        insert_at = 0
    df.insert(insert_at, 'latent_found', latent_found.values)

    # ---- Build the 512 latent columns (NaN where unmatched) ----
    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    latent_matrix = np.full((len(df), latent_dim), np.nan, dtype=np.float64)
    for row_idx, name in enumerate(df[match_col].values):
        vec = latent_by_name.get(name)
        if vec is not None:
            latent_matrix[row_idx] = vec
    latent_df = pd.DataFrame(latent_matrix, columns=latent_cols, index=df.index)

    out_df = pd.concat([df, latent_df], axis=1)

    # ---- Report ----
    n_matched = int(latent_found.sum())
    n_unmatched_rows = len(df) - n_matched
    matched_csv_names = set(df.loc[latent_found, match_col])
    latents_unused = [n for n in basenames if n not in set(df[match_col].dropna())]
    print(f'[report] CSV rows matched to a latent : {n_matched}')
    print(f'[report] CSV rows with NO latent      : {n_unmatched_rows} (latent cols left blank)')
    print(f'[report] latents not present in CSV    : {len(latents_unused)}')
    if latents_unused[:5]:
        print(f'[report]   e.g. {latents_unused[:5]}')

    # ---- Output ----
    run_name = os.path.basename(os.path.normpath(run_folder))
    ckpt = os.path.splitext(args.latent_name)[0]
    out_name = f'vtk_to_landmarks_with_latents_{run_name}_{ckpt}.csv'
    out_path = os.path.join(run_folder, out_name)
    out_df.to_csv(out_path, index=False)
    print(f'[done] wrote {len(out_df)} rows x {len(out_df.columns)} cols -> {out_path}')

    # ---- CCA analysis on the in-memory merged table ----
    if args.skip_cca:
        print('[cca] --skip_cca set; skipping CCA analysis.')
        return
    cca_out_dir = os.path.join(run_folder, CCA_SUBDIR)
    run_cca_analysis(out_df, cca_out_dir, args)


def _inv_sqrt(C, reg):
    """Symmetric positive-definite inverse square-root with ridge regularization."""
    vals, vecs = eigh(C)
    vals = np.maximum(vals, 0) + reg
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def run_cca_analysis(df, out_dir, args):
    """Non-residual CCA: LM coords (PCA-reduced) vs. raw latent dims.

    Ported from non_residual_cca_analysis.py. Operates on the in-memory merged
    DataFrame; writes 7 plots + 5 CSVs into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n[cca] ===== CCA analysis =====')
    print(f'[cca] out dir: {out_dir}')

    # ---- Column detection (same convention as the merged CSV) ----
    LM_COLS     = [c for c in df.columns if c.startswith('LM ')]
    LATENT_COLS = [c for c in df.columns if c.startswith('latent_') and c != 'latent_found']
    print(f'[cca] LM cols: {len(LM_COLS)} | latent cols: {len(LATENT_COLS)}')
    if not LM_COLS or not LATENT_COLS:
        print('[cca] ERROR: missing LM or latent columns; aborting CCA.')
        return

    # ---- Clean: drop NaN-species, then keep valid latent rows ----
    n_drop = df['species'].isna().sum()
    df = df.dropna(subset=['species']).reset_index(drop=True)
    print(f'[cca] dropped {n_drop} NaN-species rows -> {len(df)} remain')

    n_lm  = min(args.n_lm,  len(LM_COLS))
    n_lat = min(args.n_lat, len(LATENT_COLS))

    valid_mask = df['latent_found'].astype(bool).values
    print(f'[cca] valid latent rows: {valid_mask.sum()} / {len(valid_mask)}')

    lm_valid  = df[LM_COLS].values[valid_mask].astype(np.float64)
    lat_valid = df[LATENT_COLS].values[valid_mask].astype(np.float64)
    df_valid  = df[valid_mask].reset_index(drop=True)
    n = lm_valid.shape[0]
    if n < 3:
        print('[cca] ERROR: not enough valid rows for CCA; aborting.')
        return

    # ---- Dimensionality reduction ----
    pca_lm = PCA(n_components=n_lm)
    X = pca_lm.fit_transform(lm_valid).astype(np.float64)   # LM: top n_lm PCs
    Y = lat_valid[:, :n_lat].astype(np.float64)             # Latent: first n_lat raw dims
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    p, q = X.shape[1], Y.shape[1]
    K = min(p, q)
    print(f'[cca] X={X.shape} Y={Y.shape} K={K}')

    # ---- Covariances + regularized whitening + SVD ----
    Cxx = X.T @ X / (n - 1)
    Cyy = Y.T @ Y / (n - 1)
    Cxy = X.T @ Y / (n - 1)
    Cxx_inv_sqrt = _inv_sqrt(Cxx, args.reg)
    Cyy_inv_sqrt = _inv_sqrt(Cyy, args.reg)

    M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s, U, Vt = s[:K], U[:, :K], Vt[:K, :]
    canonical_corrs = np.clip(s, 0, 1)
    print(f'[cca] top-5 canonical correlations: {canonical_corrs[:5].round(4)}')

    # ---- Canonical weights & variate scores ----
    W_x = Cxx_inv_sqrt @ U           # (p, K)
    W_y = Cyy_inv_sqrt @ Vt.T        # (q, K)
    A = X @ W_x                      # (n, K) LM variate scores
    B = Y @ W_y                      # (n, K) latent variate scores

    actual_r = np.array([np.corrcoef(A[:, k], B[:, k])[0, 1] for k in range(min(5, K))])
    print(f'[cca] sanity corr(A,B) diagonal: {actual_r.round(4)}')

    # ---- Back-project LM weights to original coord space ----
    lm_loadings_orig = pca_lm.components_.T @ W_x    # (n_LM, K)

    # ---- Bartlett chi-square significance test ----
    rows = []
    for k in range(K):
        remaining = canonical_corrs[k:]
        chi_sq = -(n - 1 - (p + q + 1) / 2) * np.sum(np.log(1 - remaining ** 2))
        df_stat = (p - k) * (q - k)
        pval = chi2_dist.sf(chi_sq, df_stat)
        rows.append({'rank': k + 1, 'r': canonical_corrs[k], 'r2': canonical_corrs[k] ** 2,
                     'chi_sq': chi_sq, 'df': df_stat, 'p_value': pval,
                     'significant': pval < 0.05})
    corr_df = pd.DataFrame(rows)
    print('\n[cca] --- Canonical Correlations ---')
    print(corr_df[['rank', 'r', 'r2', 'p_value', 'significant']].to_string(index=False))

    # ---- Plot helpers ----
    families = sorted(df_valid['family'].dropna().unique())
    cmap_fam = plt.get_cmap('tab20', len(families))
    fam_to_color = {f: cmap_fam(i) for i, f in enumerate(families)}

    def _save(fig, fname):
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ---- Plot 1: canonical correlation scree ----
    n_bars = min(args.n_show * 2, K)
    fig, ax = plt.subplots(figsize=(max(8, n_bars * 0.5), 5))
    colors = ['tomato' if corr_df.loc[k, 'significant'] else 'steelblue' for k in range(n_bars)]
    ax.bar(range(1, n_bars + 1), canonical_corrs[:n_bars], color=colors, alpha=0.85)
    ax.plot(range(1, n_bars + 1), canonical_corrs[:n_bars], 'ko-', markersize=4)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Canonical Dimension')
    ax.set_ylabel('Canonical Correlation (r)')
    ax.set_title(f'Canonical Correlations  (red = p<0.05, n={n})\n'
                 f'n_lm={n_lm} PCA-PCs | n_lat={n_lat} raw dims | reg={args.reg}')
    ax.set_xticks(range(1, n_bars + 1))
    plt.tight_layout()
    _save(fig, 'cca_scree.png')

    # ---- Plot 2: canonical variate scatter grid (colored by family) ----
    n_p = min(args.n_plot, K)
    ncols = min(n_p, 3)
    nrows = (n_p + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.array(axes).flatten() if n_p > 1 else [axes]
    for k in range(n_p):
        ax = axes[k]
        for fam in families:
            mask = df_valid['family'].values == fam
            if mask.sum() == 0:
                continue
            ax.scatter(A[mask, k], B[mask, k], color=fam_to_color[fam],
                       s=10, alpha=0.6, label=fam, rasterized=True)
        r_k  = canonical_corrs[k]
        pval = corr_df.loc[k, 'p_value']
        pstr = f'{pval:.2e}' if pval < 0.001 else f'{pval:.4f}'
        m, b = np.polyfit(A[:, k], B[:, k], 1)
        xl = np.linspace(A[:, k].min(), A[:, k].max(), 200)
        ax.plot(xl, m * xl + b, 'r-', linewidth=1.2)
        ax.set_xlabel(f'LM Canonical Variate {k+1}')
        ax.set_ylabel(f'Latent Canonical Variate {k+1}')
        ax.set_title(f'CCA{k+1}  r={r_k:.4f}  p={pstr}')
    for j in range(n_p, len(axes)):
        axes[j].set_visible(False)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=fam_to_color[f], markersize=6, label=f)
               for f in families]
    fig.legend(handles=handles, fontsize=6, markerscale=1.2, ncol=3,
               bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.suptitle('Canonical Variate Pairs — colored by family', fontsize=12)
    plt.tight_layout()
    _save(fig, 'cca_scatter_grid.png')

    # ---- Plot 3: LM-PC weight heatmap ----
    n_s = min(args.n_show, K)
    fig, ax = plt.subplots(figsize=(n_s * 1.0 + 2, p * 0.35 + 1.5))
    heat_data = W_x[:, :n_s]
    vmax = np.abs(heat_data).max()
    im = ax.imshow(heat_data.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Weight')
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f'CCA{k+1}  r={canonical_corrs[k]:.3f}' for k in range(n_s)], fontsize=8)
    ax.set_xticks(range(p))
    ax.set_xticklabels([f'LM_PC{i+1}' for i in range(p)], rotation=90, fontsize=7)
    ax.set_title(f'LM Canonical Weights (n_lm={n_lm} PCA-PCs)')
    plt.tight_layout()
    _save(fig, 'cca_lm_weight_heatmap.png')

    # ---- Plot 4: latent-dim weight heatmap ----
    fig, ax = plt.subplots(figsize=(q * 0.22 + 2, n_s * 0.55 + 1.5))
    heat_data = W_y[:, :n_s].T
    vmax = np.abs(heat_data).max()
    im = ax.imshow(heat_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Weight')
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f'CCA{k+1}  r={canonical_corrs[k]:.3f}' for k in range(n_s)], fontsize=8)
    ax.set_xticks(range(0, q, max(1, q // 20)))
    ax.set_xticklabels([f'lat_{i}' for i in range(0, q, max(1, q // 20))], rotation=90, fontsize=7)
    ax.set_title(f'Latent Canonical Weights (n_lat={n_lat} raw dims)')
    plt.tight_layout()
    _save(fig, 'cca_lat_weight_heatmap.png')

    # ---- Plot 5: top-15 LM original-space loadings for CCA1-4 ----
    TOP_N = 15
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, k in enumerate(range(min(4, K))):
        ax = axes[idx]
        loadings = lm_loadings_orig[:, k]
        top_idx  = np.argsort(np.abs(loadings))[-TOP_N:][::-1]
        top_vals = loadings[top_idx]
        top_lbls = [LM_COLS[i] for i in top_idx]
        colors_bar = ['tomato' if v > 0 else 'steelblue' for v in top_vals]
        ax.barh(range(TOP_N), top_vals[::-1], color=colors_bar[::-1], alpha=0.85)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(top_lbls[::-1], fontsize=8)
        ax.axvline(0, color='k', linewidth=0.7)
        ax.set_xlabel('Back-projected weight')
        ax.set_title(f'CCA{k+1} — Top {TOP_N} LM original-space loadings  (r={canonical_corrs[k]:.4f})')
    plt.suptitle('LM Original-Space Loadings: CCA1–CCA4', fontsize=13)
    plt.tight_layout()
    _save(fig, 'cca_lm_original_loadings_top15.png')

    # ---- Plot 6: top-15 latent-dim loadings for CCA1-4 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, k in enumerate(range(min(4, K))):
        ax = axes[idx]
        weights  = W_y[:, k]
        top_idx  = np.argsort(np.abs(weights))[-TOP_N:][::-1]
        top_vals = weights[top_idx]
        top_lbls = [LATENT_COLS[i] for i in top_idx]
        colors_bar = ['tomato' if v > 0 else 'steelblue' for v in top_vals]
        ax.barh(range(TOP_N), top_vals[::-1], color=colors_bar[::-1], alpha=0.85)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(top_lbls[::-1], fontsize=8)
        ax.axvline(0, color='k', linewidth=0.7)
        ax.set_xlabel('Weight')
        ax.set_title(f'CCA{k+1} — Top {TOP_N} raw latent dims  (r={canonical_corrs[k]:.4f})')
    plt.suptitle('Latent Canonical Weights: CCA1–CCA4', fontsize=13)
    plt.tight_layout()
    _save(fig, 'cca_lat_original_loadings_top15.png')

    # ---- Plot 7: cross-variate correlation matrix ----
    n_cv = min(args.n_show, K)
    cross_r = np.zeros((n_cv, n_cv))
    for i in range(n_cv):
        for j in range(n_cv):
            cross_r[i, j] = np.corrcoef(A[:, i], B[:, j])[0, 1]
    fig, ax = plt.subplots(figsize=(n_cv + 1.5, n_cv + 1))
    im = ax.imshow(cross_r, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(n_cv)); ax.set_xticklabels([f'Lat_CV{j+1}' for j in range(n_cv)], rotation=45, ha='right')
    ax.set_yticks(range(n_cv)); ax.set_yticklabels([f'LM_CV{i+1}' for i in range(n_cv)])
    for i in range(n_cv):
        for j in range(n_cv):
            ax.text(j, i, f'{cross_r[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(cross_r[i, j]) < 0.6 else 'white')
    ax.set_title('Cross-Variate Correlations (should be ~diagonal)')
    plt.tight_layout()
    _save(fig, 'cca_cross_variate_heatmap.png')

    # ---- CSV outputs ----
    corr_df.to_csv(os.path.join(out_dir, 'cca_canonical_correlations.csv'), index=False)
    pd.DataFrame(W_x, index=[f'LM_PC{i+1}' for i in range(p)],
                 columns=[f'CCA{k+1}' for k in range(K)]
                 ).to_csv(os.path.join(out_dir, 'cca_lm_weights.csv'))
    pd.DataFrame(W_y, index=LATENT_COLS[:q],
                 columns=[f'CCA{k+1}' for k in range(K)]
                 ).to_csv(os.path.join(out_dir, 'cca_lat_weights.csv'))
    pd.DataFrame(lm_loadings_orig, index=LM_COLS,
                 columns=[f'CCA{k+1}' for k in range(K)]
                 ).to_csv(os.path.join(out_dir, 'cca_lm_loadings_original.csv'))
    score_dict = {'species': df_valid['species'].values, 'family': df_valid['family'].values}
    for k in range(K):
        score_dict[f'A_{k+1}'] = A[:, k]
        score_dict[f'B_{k+1}'] = B[:, k]
    pd.DataFrame(score_dict).to_csv(os.path.join(out_dir, 'cca_variate_scores.csv'), index=False)

    print(f'[cca] all outputs saved to: {out_dir}')


if __name__ == '__main__':
    main()
