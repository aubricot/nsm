#!/usr/bin/env python3
# visualize_latent_space.py
# ========================
# Generates PCA, t-SNE, and UMAP visualizations of latent spaces from
# previously trained NSM models or ablation classification runs.
#
# Usage examples:
#   # Basic: visualize a model run
#   python visualize_latent_space.py --run-dir run_v57
#
#   # Highlight two species in all plots
#   python visualize_latent_space.py --run-dir run_v57 --highlight "Cordylidae_Ouroborus_cataphractus" "Scincidae_Chalcides_ocellatus"
#
#   # Use a specific checkpoint
#   python visualize_latent_space.py --run-dir run_v57 --checkpoint 2000
#
#   # Color by family, genus, species, position, or life_history
#   python visualize_latent_space.py --run-dir run_v57 --color-by family
#
#   # Point to an ablation study to overlay test predictions
#   python visualize_latent_space.py --run-dir run_v57 --ablation-dir results/ablation_run_v57_ckpt3000_20260319_120000
#
#   # Customize t-SNE and UMAP parameters
#   python visualize_latent_space.py --run-dir run_v57 --tsne-perplexity 30 --umap-neighbors 50
#
# Output:
#   results/latent_viz_YYYYMMDD_HHMMSS/
#   ├── README.md
#   ├── pca_1v2.png
#   ├── pca_3v4.png
#   ├── tsne.png
#   ├── umap.png
#   ├── pca_1v2_highlighted.png       (if --highlight used)
#   ├── pca_3v4_highlighted.png
#   ├── tsne_highlighted.png
#   ├── umap_highlighted.png
#   ├── tsne_kde_life_history.png
#   ├── coordinates.csv               (all reduced coords for external analysis)
#   └── highlight_detail.csv           (if --highlight used)
# ========================

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import sys
import argparse
import glob as glob_module
import json
import re
import datetime
import colorsys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .taxonomy import parse_taxonomy_from_filename
from .run_utils import create_run_directory, write_run_manifest


# ======================================================================
# CLI Arguments
# ======================================================================
def _detect_latest_checkpoint(run_dir: str) -> str:
    """Find the highest-numbered checkpoint in run_dir/model/."""
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(model_dir):
        # Try latent_codes dir as fallback
        model_dir = os.path.join(run_dir, "latent_codes")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No model or latent_codes directory found in {run_dir}")
    pth_files = glob_module.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")
    epochs = []
    for p in pth_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.isdigit():
            epochs.append(int(stem))
    if not epochs:
        raise ValueError(f"No numerically-named checkpoints in {model_dir}")
    return str(max(epochs))


parser = argparse.ArgumentParser(
    description="Visualize NSM latent space with PCA, t-SNE, and UMAP.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python visualize_latent_space.py --run-dir run_v57
  python visualize_latent_space.py --run-dir run_v57 --highlight "Cordylidae_Ouroborus_cataphractus" "Scincidae_Chalcides_ocellatus"
  python visualize_latent_space.py --run-dir run_v57 --color-by family --tsne-perplexity 30
  python visualize_latent_space.py --run-dir run_v57 --ablation-dir results/ablation_run_v57_ckpt3000_20260319
""",
)
parser.add_argument(
    "--run-dir", type=str, required=True,
    help="Training run directory containing model_params_config.json and latent_codes/",
)
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Checkpoint epoch (e.g. '2000'). If omitted, uses the latest.",
)
parser.add_argument(
    "--highlight", nargs="+", default=None,
    help="Species to highlight (substring match on filename). At least 2 recommended. "
         "E.g. --highlight ouroborus chalcides",
)
parser.add_argument(
    "--color-by", type=str, default="species",
    choices=["species", "genus", "family", "position", "life_history"],
    help="Attribute to color points by (default: species).",
)
parser.add_argument(
    "--ablation-dir", type=str, default=None,
    help="Path to an ablation results directory to overlay test-set predictions.",
)
parser.add_argument(
    "--tsne-perplexity", type=float, default=30,
    help="t-SNE perplexity (default: 30).",
)
parser.add_argument(
    "--tsne-metric", type=str, default="cosine",
    help="t-SNE distance metric (default: cosine).",
)
parser.add_argument(
    "--umap-neighbors", type=int, default=50,
    help="UMAP n_neighbors (default: 50).",
)
parser.add_argument(
    "--umap-min-dist", type=float, default=0.2,
    help="UMAP min_dist (default: 0.2).",
)
parser.add_argument(
    "--no-kde", action="store_true",
    help="Disable KDE density clouds on t-SNE life history plot.",
)
parser.add_argument(
    "--dpi", type=int, default=300,
    help="Output image DPI (default: 300).",
)
args = parser.parse_args()


# ======================================================================
# Life history strategy functions (from notebook)
# ======================================================================
def get_marker(species: str) -> str:
    """Map species to marker based on life history strategy."""
    species = species.lower()
    if "ouroborus" in species:
        return 'v'
    elif any(k in species for k in ["chalcides", "tetradactylus", "chamaesaura"]):
        return 'P'
    elif any(k in species for k in ["skoog", "eremiascincus", "_scincus"]):
        return '+'
    elif any(k in species for k in ["acontias", "mochlus", "rhineura", "dibamus", "lanthonotus", "bipes", "diplometopon", "pseudopus"]):
        return 's'
    elif any(k in species for k in ["jonesi", "corucia", "gecko", "chamaeleo", "iguana", "brookesia", "dracaena", "anolis", "basiliscus",
                                    "aristelliger", "sceloporus", "lialis", "phyllurus"]):
        return 'd'
    elif any(k in species for k in ["elgaria", "smaug_giganteus", "broadleysaurus", "ateuchosaurus", "alopoglossus", "heloderma", "tupinambis",
                                    "carlia", "lipinia", "tiliqua", "tribolonotus", "leiolepis", "eublepharis", "oreosaurus", "baranus",
                                    "callopistes", "cricosaura", "lepidophyma", "sphenodon", "lacerta", "enyaloides", "crocodilurus"]):
        return 'X'
    elif any(k in species for k in ["eryx", "homalopsis", "aniolios"]):
        return '2'
    else:
        return 'o'


LEGEND_ITEMS = [
    ('v', 'Bites tail to side'),
    ('P', 'Grass swimmer'),
    ('+', 'Sand swimmer'),
    ('s', 'Burrowers'),
    ('d', 'Arboreal'),
    ('X', 'Terrestrial'),
    ('o', 'Saxicolous'),
    ('2', 'Snake'),
]

MARKER_COLORS = {
    'v': (0.32, 0.24, 0.56),
    'P': (0.65, 0.69, 0.12),
    '+': (0.84, 0.65, 0.23),
    's': (0.72, 0.44, 0.22),
    'd': (0.36, 0.557, 0.68),
    'X': (0.10, 0.51, 0.40),
    'o': (0.60, 0.50, 0.46),
    '2': (0, 0, 0),
}


def extract_family_from_species(species_label: str) -> str:
    """Extract family group from species label (from notebook)."""
    family = species_label.split('_')[0]
    sl = species_label.lower()
    if any(k in sl for k in ["scincus", "scincidae"]):
        family = "Scincidae"
    elif "ouroborus" in sl:
        family = "Cordylidae_Ouroborus"
    elif any(k in sl for k in ["gerrhosaurus", "gerrho"]):
        family = "Gerrhosauridae"
    elif any(k in sl for k in ["chameleo", "iguana", "agamidae", "anolidae", "corytophanidae",
                                "crotaphytidae", "hoplocercidae", "leiocephalidae", "leiosauridae",
                                "phrynosomatidae", "tropiduridae"]):
        family = "Iguania"
    elif any(k in sl for k in ["anguidae", "lanthonotus", "varanus", "shinosaurus", "heloderma"]):
        family = "Anguimorpha"
    elif any(k in sl for k in ["lacertidae", "lacerta"]):
        family = "Lacertidae"
    elif any(k in sl for k in ["bipes", "rhineura", "diplometopon"]):
        family = "Amphisbaenea"
    elif any(k in sl for k in ["gecko", "tarentola", "eublepharis", "aristelliger", "phyllurus", "lialis"]):
        family = "Gekkota"
    elif any(k in sl for k in ["eryx", "homalopsis", "aniolios"]):
        family = "Snake"
    elif "xantusiidae" in sl:
        family = "Xantusiidae"
    elif any(k in sl for k in ["gymnopthalmidae", "teiidae"]):
        family = "Gymnophthalmoidea"
    elif "dibamus" in sl:
        family = "Dibamidae"
    elif "sphenodon" in sl:
        family = "Rhynchocephalia"
    return family


FAMILY_BASE_COLORS = {
    "iguania": np.array([0.78, 0.16, 0.16]),
    "anguimorpha": np.array([1.00, 0.79, 0.20]),
    "cordylidae": np.array([0.10, 0.51, 0.40]),
    "cordylidae_ouroborus": np.array([0.082, 0.76, 0.92]),
    "gekkota": np.array([0.41, 0.227, 0.6]),
    "gymnophthalmoidea": np.array([0.73, 0.14, 0.5]),
    "gerrhosauridae": np.array([0.98, 0.39, 0.14]),
    "scincidae": np.array([0.65, 0.69, 0.12]),
    "amphisbaenea": np.array([0.60, 0.50, 0.46]),
    "lacertidae": np.array([0.145, 0.39, 0.075]),
    "xantusiidae": np.array([0.88, 0.74, 0.59]),
}
DEFAULT_FAMILY_COLOR = np.array([0.52, 0.52, 0.52])


def make_family_cmap(families: List[str]) -> Dict[str, np.ndarray]:
    """Map family names to base RGB colors."""
    family_colors = {}
    for family in families:
        family_colors[family] = FAMILY_BASE_COLORS.get(family.lower(), DEFAULT_FAMILY_COLOR)
    return family_colors


def generate_species_cmap_gradient(family_base_colors: Dict, species_groups: Dict, max_shift: float = 0.4) -> Dict[str, Tuple]:
    """Generate per-species colors as gradients within family base color."""
    species_colors = {}
    family_species_map = defaultdict(list)
    for species in species_groups:
        family = extract_family_from_species(species)
        family_species_map[family].append(species)
    for family, species_list in family_species_map.items():
        base_rgb = family_base_colors.get(family, np.array([0.7, 0.7, 0.7]))
        base_hls = colorsys.rgb_to_hls(*base_rgb)
        sorted_species = sorted(species_list)
        n = len(sorted_species)
        center_idx = n // 2
        for i, sp in enumerate(sorted_species):
            if i == center_idx:
                new_rgb = base_rgb
            else:
                shift_direction = -1 if i < center_idx else 1
                shift_amount = (abs(i - center_idx) / max(n - 1, 1)) * max_shift
                new_lightness = np.clip(base_hls[1] + shift_direction * shift_amount, 0, 1)
                new_rgb = colorsys.hls_to_rgb(base_hls[0], new_lightness, base_hls[2])
            species_colors[sp] = tuple(np.clip(new_rgb, 0, 1)) + (1.0,)
    return species_colors


def get_gradient_cmap(color, n=256, white_level=0.8):
    """Custom cmap from light to solid color for KDE plots."""
    r, g, b = color
    light_color = (
        r + (1 - r) * white_level,
        g + (1 - g) * white_level,
        b + (1 - b) * white_level,
    )
    colors = [light_color, (r, g, b)]
    return LinearSegmentedColormap.from_list("gradient_cmap", colors, N=n)


def sort_key(item):
    """Sort vertebrae by region (C < T < L) then numeric part."""
    region_order = {'C': 0, 'T': 1, 'L': 2}
    v = item[0]
    if v and len(v) > 1 and v[0] in region_order:
        try:
            return (region_order[v[0]], int(v[1:]))
        except ValueError:
            pass
    return (99, 0)


# ======================================================================
# Color assignment logic
# ======================================================================
def _assign_colors(species_groups: Dict, color_by: str, labels_parsed: List[Dict]) -> Tuple[Dict, str]:
    """
    Assign colors to each species/group key based on color_by mode.
    Returns (color_dict, legend_title).
    """
    if color_by == "life_history":
        # Use marker-based coloring (life history strategy)
        colors = {}
        for sp in species_groups:
            m = get_marker(sp)
            colors[sp] = MARKER_COLORS.get(m, (0.5, 0.5, 0.5))
        return colors, "Life History"

    if color_by == "species":
        families = sorted({extract_family_from_species(sp) for sp in species_groups})
        family_colors = make_family_cmap(families)
        sp_colors = generate_species_cmap_gradient(family_colors, species_groups)
        return sp_colors, "Species (family gradient)"

    # For family, genus, position: build a distinct color per group
    group_set = set()
    species_to_group = {}
    for sp in species_groups:
        if color_by == "family":
            group = extract_family_from_species(sp)
        elif color_by == "genus":
            # Extract genus from species label (second underscore-separated part)
            parts = sp.split('_')
            group = parts[1] if len(parts) >= 2 else sp
        elif color_by == "position":
            group = "mixed"  # position varies per data point, handled differently
        else:
            group = sp
        species_to_group[sp] = group
        group_set.add(group)

    if color_by == "family":
        families = sorted(group_set)
        family_colors = make_family_cmap(families)
        colors = {}
        for sp in species_groups:
            fam = extract_family_from_species(sp)
            colors[sp] = tuple(family_colors.get(fam, DEFAULT_FAMILY_COLOR))
        return colors, "Family"

    # Generic distinct colors for genus or other
    sorted_groups = sorted(group_set)
    cmap = plt.cm.get_cmap('tab20', max(len(sorted_groups), 1))
    group_to_color = {g: cmap(i) for i, g in enumerate(sorted_groups)}
    colors = {sp: group_to_color[species_to_group[sp]] for sp in species_groups}
    legend_title = color_by.capitalize()
    return colors, legend_title


# ======================================================================
# Plotting functions
# ======================================================================
def _plot_scatter(coords_2d: np.ndarray, species_groups: Dict, species_colors: Dict,
                  xlabel: str, ylabel: str, title: str, outpath: str,
                  highlight_species: Optional[List[str]] = None,
                  annotate_vertebra: bool = False, point_size: int = 30,
                  legend_title: str = "Species"):
    """Generic 2D scatter plot with optional species highlighting."""
    fig, ax = plt.subplots(figsize=(20, 16))

    # Determine which species are highlighted
    highlight_set = set()
    if highlight_species:
        for sp in species_groups:
            for h in highlight_species:
                if h.lower() in sp.lower():
                    highlight_set.add(sp)

    # Plot non-highlighted species first (faded)
    for species, points in species_groups.items():
        if highlight_species and species not in highlight_set:
            points_sorted = sorted(points, key=sort_key)
            verts, x_vals, y_vals = zip(*points_sorted)
            color = species_colors.get(species, (0.7, 0.7, 0.7, 1.0))
            alpha = 0.15 if highlight_species else 0.8
            ax.scatter(x_vals, y_vals, marker=get_marker(species),
                       color=color, s=point_size * 0.5, alpha=alpha, linewidths=0.3)

    # Plot highlighted species (or all if no highlighting)
    legend_handles = []
    legend_labels = []
    for species, points in species_groups.items():
        if highlight_species and species not in highlight_set:
            continue
        points_sorted = sorted(points, key=sort_key)
        verts, x_vals, y_vals = zip(*points_sorted)
        color = species_colors.get(species, (0.7, 0.7, 0.7, 1.0))
        marker = get_marker(species)
        size = point_size * 2.5 if highlight_species else point_size
        edgecolor = 'black' if highlight_species else 'none'
        linewidth = 1.5 if highlight_species else 0.5
        ax.scatter(x_vals, y_vals, marker=marker, color=color, s=size,
                   edgecolors=edgecolor, linewidths=linewidth, zorder=10,
                   label=species)

        if annotate_vertebra:
            for x, y, label in zip(x_vals, y_vals, verts):
                ax.annotate(label, (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=7, color=color)

        legend_handles.append(plt.Line2D([0], [0], marker=marker, linestyle='None',
                                          markerfacecolor=color, markeredgecolor='black' if highlight_species else color,
                                          markersize=10))
        legend_labels.append(species)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if len(legend_labels) <= 60:
        ax.legend(handles=legend_handles, labels=legend_labels,
                  loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=7, title=legend_title, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


def _plot_tsne_kde_life_history(tsne_coords: np.ndarray, labels: List[Tuple],
                                 outpath: str, show_kde: bool = True):
    """t-SNE plot colored by life history strategy with KDE clouds (from notebook)."""
    strategy_groups = defaultdict(list)
    for i, (species, vertebra) in enumerate(labels):
        if species is not None and vertebra is not None:
            strategy = get_marker(species)
            strategy_groups[strategy].append((vertebra, tsne_coords[i, 0], tsne_coords[i, 1]))

    fig, ax = plt.subplots(figsize=(20, 16))

    for strategy, points in strategy_groups.items():
        points_sorted = sorted(points, key=lambda x: (x[0], x[1]))
        verts, x_vals, y_vals = zip(*points_sorted)

        if show_kde and len(x_vals) >= 5:
            cmap = get_gradient_cmap(MARKER_COLORS[strategy], white_level=0.7)
            try:
                sns.kdeplot(x=x_vals, y=y_vals, fill=True, cmap=cmap,
                            alpha=0.3, bw_adjust=1.9, cut=5.0, thresh=0.7, levels=4, ax=ax)
            except Exception:
                pass  # KDE can fail with degenerate data

        ax.scatter(x_vals, y_vals, marker=strategy, color=MARKER_COLORS[strategy], s=15)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title("t-SNE by Life History Strategy with Density Clouds", fontsize=14)

    handles = [plt.Line2D([0], [0], marker=m, linestyle='None',
                           markerfacecolor=MARKER_COLORS[m], markeredgecolor=MARKER_COLORS[m],
                           markersize=10) for m, _ in LEGEND_ITEMS]
    llabels = [label for _, label in LEGEND_ITEMS]
    ax.legend(handles=handles, labels=llabels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(outpath, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


# ======================================================================
# Load data
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, args.run_dir)
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Run directory not found: {TRAIN_DIR}")

os.chdir(TRAIN_DIR)

CKPT = args.checkpoint if args.checkpoint else _detect_latest_checkpoint(".")
LC_PATH = f'latent_codes/{CKPT}.pth'
if not os.path.isfile(LC_PATH):
    raise FileNotFoundError(f"Latent codes not found at {LC_PATH}")

RUN_NAME = os.path.basename(args.run_dir)
print(f"Run directory: {RUN_NAME}")
print(f"Checkpoint:    {CKPT}")

# Load config
config_path = 'model_params_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load latent codes
latent_ckpt = torch.load(LC_PATH, map_location="cpu", weights_only=False)
codes = latent_ckpt['latent_codes']['weight'].detach().cpu().numpy()
print(f"Loaded {codes.shape[0]} latent codes of dimension {codes.shape[1]}")

# Parse labels from filenames using both the notebook regex and taxonomy parser
pat = re.compile(r"^(?P<species>[\w\s\-]+)[\-_ ]+\d+[\-_ ]+(?P<vertebra>[CTL]\d+)", re.IGNORECASE)
labels = []  # (species_label, vertebra_str) tuples
labels_parsed = []  # taxonomy dicts
for f in all_vtk_files:
    fname = os.path.basename(f)
    m = pat.match(fname)
    if m:
        species = m.group("species").strip()
        vertebra = m.group("vertebra").strip()
        labels.append((species, vertebra))
    else:
        labels.append((None, None))

    parsed = parse_taxonomy_from_filename(fname)
    labels_parsed.append(parsed)

print(f"Parsed labels for {sum(1 for s, v in labels if s is not None)} / {len(labels)} files")

# Report highlight matches
if args.highlight:
    for h in args.highlight:
        matches = [s for s, v in labels if s and h.lower() in s.lower()]
        unique = sorted(set(matches))
        print(f"  Highlight '{h}': {len(matches)} data points across {len(unique)} species: {unique[:5]}{'...' if len(unique) > 5 else ''}")


# ======================================================================
# Compute dimensionality reductions
# ======================================================================
print("\nComputing dimensionality reductions...")

# PCA (4 components)
print("  PCA...")
pca = PCA(n_components=4)
pca_coords = pca.fit_transform(codes)
pca_var = pca.explained_variance_ratio_

# t-SNE
print(f"  t-SNE (perplexity={args.tsne_perplexity}, metric={args.tsne_metric})...")
tsne = TSNE(
    n_components=2,
    perplexity=args.tsne_perplexity,
    learning_rate=200,
    early_exaggeration=12,
    n_iter_without_progress=2000,
    metric=args.tsne_metric,
    random_state=42,
)
tsne_coords = tsne.fit_transform(codes)

# UMAP (PCA pre-reduction to 50 dims, as in notebook)
print(f"  UMAP (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
try:
    import umap.umap_ as umap_lib
    pca_pre = PCA(n_components=min(50, codes.shape[1]))
    codes_pca = pca_pre.fit_transform(codes)
    umap_reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        spread=0.5,
        n_epochs=500,
        random_state=42,
    )
    umap_coords = umap_reducer.fit_transform(codes_pca)
    umap_available = True
except ImportError:
    print("  Warning: umap-learn not installed. Skipping UMAP plots.")
    print("  Install with: pip install umap-learn")
    umap_coords = None
    umap_available = False

print("  Done.")


# ======================================================================
# Build species_groups structure for plotting
# ======================================================================
def _build_species_groups(coords_2d: np.ndarray) -> Dict[str, List[Tuple]]:
    """Group 2D coordinates by species, storing (vertebra, x, y)."""
    groups = defaultdict(list)
    for i, (species, vertebra) in enumerate(labels):
        if species is not None and vertebra is not None:
            groups[species].append((vertebra, coords_2d[i, 0], coords_2d[i, 1]))
    return dict(groups)


# ======================================================================
# Create output directory
# ======================================================================
out_dir = create_run_directory(base_dir="results", prefix=f"latent_viz_{RUN_NAME}_ckpt{CKPT}")
print(f"\nOutput directory: {out_dir}")


# ======================================================================
# Generate all plots
# ======================================================================
print("\nGenerating plots...")

# Build color maps
pca_12_groups = _build_species_groups(pca_coords[:, :2])
species_colors, legend_title = _assign_colors(pca_12_groups, args.color_by, labels_parsed)

# --- PCA 1 vs 2 ---
_plot_scatter(
    pca_coords[:, :2], pca_12_groups, species_colors,
    xlabel=f"PC1: {pca_var[0]*100:.2f}%", ylabel=f"PC2: {pca_var[1]*100:.2f}%",
    title=f"PCA (PC1 vs PC2) — {RUN_NAME} ckpt {CKPT}",
    outpath=os.path.join(out_dir, "pca_1v2.png"),
    annotate_vertebra=True, point_size=40, legend_title=legend_title,
)

# --- PCA 3 vs 4 ---
pca_34_groups = _build_species_groups(pca_coords[:, 2:4])
species_colors_34, _ = _assign_colors(pca_34_groups, args.color_by, labels_parsed)
_plot_scatter(
    pca_coords[:, 2:4], pca_34_groups, species_colors_34,
    xlabel=f"PC3: {pca_var[2]*100:.2f}%", ylabel=f"PC4: {pca_var[3]*100:.2f}%",
    title=f"PCA (PC3 vs PC4) — {RUN_NAME} ckpt {CKPT}",
    outpath=os.path.join(out_dir, "pca_3v4.png"),
    annotate_vertebra=True, point_size=40, legend_title=legend_title,
)

# --- t-SNE ---
tsne_groups = _build_species_groups(tsne_coords)
species_colors_tsne, _ = _assign_colors(tsne_groups, args.color_by, labels_parsed)
_plot_scatter(
    tsne_coords, tsne_groups, species_colors_tsne,
    xlabel="t-SNE 1", ylabel="t-SNE 2",
    title=f"t-SNE — {RUN_NAME} ckpt {CKPT} (perplexity={args.tsne_perplexity})",
    outpath=os.path.join(out_dir, "tsne.png"),
    point_size=20, legend_title=legend_title,
)

# --- t-SNE KDE (life history) ---
_plot_tsne_kde_life_history(
    tsne_coords, labels,
    outpath=os.path.join(out_dir, "tsne_kde_life_history.png"),
    show_kde=not args.no_kde,
)

# --- UMAP ---
if umap_available:
    umap_groups = _build_species_groups(umap_coords)
    species_colors_umap, _ = _assign_colors(umap_groups, args.color_by, labels_parsed)
    _plot_scatter(
        umap_coords, umap_groups, species_colors_umap,
        xlabel="UMAP 1", ylabel="UMAP 2",
        title=f"UMAP — {RUN_NAME} ckpt {CKPT} (neighbors={args.umap_neighbors})",
        outpath=os.path.join(out_dir, "umap.png"),
        annotate_vertebra=True, point_size=40, legend_title=legend_title,
    )


# ======================================================================
# Highlighted species plots (if --highlight provided)
# ======================================================================
if args.highlight:
    print("\nGenerating highlighted plots...")

    # PCA 1v2 highlighted
    _plot_scatter(
        pca_coords[:, :2], pca_12_groups, species_colors,
        xlabel=f"PC1: {pca_var[0]*100:.2f}%", ylabel=f"PC2: {pca_var[1]*100:.2f}%",
        title=f"PCA (PC1 vs PC2) — Highlighted — {RUN_NAME} ckpt {CKPT}",
        outpath=os.path.join(out_dir, "pca_1v2_highlighted.png"),
        highlight_species=args.highlight, annotate_vertebra=True,
        point_size=40, legend_title="Highlighted Species",
    )

    # PCA 3v4 highlighted
    _plot_scatter(
        pca_coords[:, 2:4], pca_34_groups, species_colors_34,
        xlabel=f"PC3: {pca_var[2]*100:.2f}%", ylabel=f"PC4: {pca_var[3]*100:.2f}%",
        title=f"PCA (PC3 vs PC4) — Highlighted — {RUN_NAME} ckpt {CKPT}",
        outpath=os.path.join(out_dir, "pca_3v4_highlighted.png"),
        highlight_species=args.highlight, annotate_vertebra=True,
        point_size=40, legend_title="Highlighted Species",
    )

    # t-SNE highlighted
    _plot_scatter(
        tsne_coords, tsne_groups, species_colors_tsne,
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        title=f"t-SNE — Highlighted — {RUN_NAME} ckpt {CKPT}",
        outpath=os.path.join(out_dir, "tsne_highlighted.png"),
        highlight_species=args.highlight, point_size=30,
        legend_title="Highlighted Species",
    )

    # UMAP highlighted
    if umap_available:
        _plot_scatter(
            umap_coords, umap_groups, species_colors_umap,
            xlabel="UMAP 1", ylabel="UMAP 2",
            title=f"UMAP — Highlighted — {RUN_NAME} ckpt {CKPT}",
            outpath=os.path.join(out_dir, "umap_highlighted.png"),
            highlight_species=args.highlight, annotate_vertebra=True,
            point_size=40, legend_title="Highlighted Species",
        )

    # Export highlight detail CSV
    highlight_rows = []
    for i, (species, vertebra) in enumerate(labels):
        if species is None:
            continue
        is_highlighted = any(h.lower() in species.lower() for h in args.highlight)
        if is_highlighted:
            row = {
                "filename": all_vtk_files[i],
                "species": species,
                "vertebra": vertebra,
                "family": extract_family_from_species(species),
                "life_history": dict(LEGEND_ITEMS).get(get_marker(species), "Unknown"),
                "PC1": pca_coords[i, 0], "PC2": pca_coords[i, 1],
                "PC3": pca_coords[i, 2], "PC4": pca_coords[i, 3],
                "tSNE1": tsne_coords[i, 0], "tSNE2": tsne_coords[i, 1],
            }
            if umap_available:
                row["UMAP1"] = umap_coords[i, 0]
                row["UMAP2"] = umap_coords[i, 1]
            highlight_rows.append(row)

    if highlight_rows:
        highlight_df = pd.DataFrame(highlight_rows)
        highlight_path = os.path.join(out_dir, "highlight_detail.csv")
        highlight_df.to_csv(highlight_path, index=False)
        print(f"  Highlight detail: {len(highlight_rows)} points across {highlight_df['species'].nunique()} species")


# ======================================================================
# Export all coordinates for external analysis
# ======================================================================
print("\nExporting coordinates...")
coord_rows = []
for i, (species, vertebra) in enumerate(labels):
    if species is None:
        continue
    parsed = labels_parsed[i]
    row = {
        "filename": all_vtk_files[i],
        "species": species,
        "vertebra": vertebra,
        "family": extract_family_from_species(species),
        "genus": parsed.get('genus') if parsed else None,
        "taxonomy_species": parsed.get('species') if parsed else None,
        "position": parsed.get('position') if parsed else None,
        "life_history": dict(LEGEND_ITEMS).get(get_marker(species), "Unknown"),
        "PC1": pca_coords[i, 0], "PC2": pca_coords[i, 1],
        "PC3": pca_coords[i, 2], "PC4": pca_coords[i, 3],
        "tSNE1": tsne_coords[i, 0], "tSNE2": tsne_coords[i, 1],
    }
    if umap_available:
        row["UMAP1"] = umap_coords[i, 0]
        row["UMAP2"] = umap_coords[i, 1]
    coord_rows.append(row)

coords_df = pd.DataFrame(coord_rows)
coords_path = os.path.join(out_dir, "coordinates.csv")
coords_df.to_csv(coords_path, index=False)
print(f"  Exported {len(coord_rows)} coordinate rows")


# ======================================================================
# Overlay ablation test predictions (if --ablation-dir provided)
# ======================================================================
if args.ablation_dir:
    ablation_path = args.ablation_dir
    if not os.path.isabs(ablation_path):
        ablation_path = os.path.join(TRAIN_DIR, ablation_path)

    if os.path.isdir(ablation_path):
        print(f"\nOverlaying ablation predictions from: {ablation_path}")
        # Find prediction CSVs across configs
        for cfg_dir_name in sorted(os.listdir(ablation_path)):
            pred_csv = os.path.join(ablation_path, cfg_dir_name, "predictions.csv")
            if not os.path.isfile(pred_csv):
                continue
            pred_df = pd.read_csv(pred_csv)
            if "cos_top1_species_match" not in pred_df.columns and "cos_top1_match" not in pred_df.columns:
                continue

            # Determine correct/incorrect per test mesh for cosine top-1
            match_col = "cos_top1_species_match" if "cos_top1_species_match" in pred_df.columns else "cos_top1_match"
            correct = pred_df[match_col] == "yes"

            # Report
            acc = correct.mean()
            print(f"  {cfg_dir_name}: Cosine Top-1 = {acc:.1%} ({correct.sum()}/{len(correct)})")
    else:
        print(f"Warning: Ablation directory not found: {ablation_path}")


# ======================================================================
# Write manifest
# ======================================================================
produced_files = {
    "pca_1v2.png": "PCA plot (PC1 vs PC2) colored by " + args.color_by,
    "pca_3v4.png": "PCA plot (PC3 vs PC4) colored by " + args.color_by,
    "tsne.png": f"t-SNE plot (perplexity={args.tsne_perplexity}, metric={args.tsne_metric})",
    "tsne_kde_life_history.png": "t-SNE with KDE density clouds by life history strategy",
    "coordinates.csv": "All reduced coordinates (PCA, t-SNE, UMAP) with taxonomy metadata",
}
if umap_available:
    produced_files["umap.png"] = f"UMAP plot (neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})"
if args.highlight:
    produced_files["pca_1v2_highlighted.png"] = "PCA (PC1 vs PC2) with highlighted species"
    produced_files["pca_3v4_highlighted.png"] = "PCA (PC3 vs PC4) with highlighted species"
    produced_files["tsne_highlighted.png"] = "t-SNE with highlighted species"
    if umap_available:
        produced_files["umap_highlighted.png"] = "UMAP with highlighted species"
    produced_files["highlight_detail.csv"] = "Coordinates for highlighted species only"

highlight_note = ""
if args.highlight:
    highlight_note = f"\n\nHighlighted species (substring match): {', '.join(args.highlight)}"

write_run_manifest(
    out_dir,
    description=(
        f"Latent space visualization for {RUN_NAME} checkpoint {CKPT}. "
        f"Colored by {args.color_by}. "
        f"{codes.shape[0]} latent codes of dimension {codes.shape[1]}."
    ),
    approach=f"PCA + t-SNE + UMAP visualization (color by {args.color_by})",
    script_path=os.path.abspath(__file__),
    checkpoint=CKPT,
    notes=(
        f"Model: {RUN_NAME} (checkpoint {CKPT})\n"
        f"Latent codes: {codes.shape[0]} x {codes.shape[1]}\n"
        f"Color mode: {args.color_by}\n"
        f"t-SNE: perplexity={args.tsne_perplexity}, metric={args.tsne_metric}\n"
        f"UMAP: n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}\n"
        f"PCA explained variance: PC1={pca_var[0]*100:.2f}%, PC2={pca_var[1]*100:.2f}%, "
        f"PC3={pca_var[2]*100:.2f}%, PC4={pca_var[3]*100:.2f}%"
        + highlight_note
    ),
    extra_files=produced_files,
)

print(f"\n{'='*60}")
print(f"Visualization complete! All outputs saved to:")
print(f"  {out_dir}")
print(f"{'='*60}")
