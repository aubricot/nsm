"""
Compute and save mean latent vectors per taxonomic group (family, genus, species).

This is a prerequisite for classify_vertebrae_normpos.py when using
--init family/genus/species. It reads the trained model's latent codes,
groups them by taxonomy parsed from the VTK filenames, and saves the per-group
mean latent vectors for use as initialization during novel mesh classification.

Usage:
    python create_latents_per_group.py
    python create_latents_per_group.py --run_name run_v57a

Arguments:
    --run_name TEXT
        Training directory name containing model checkpoint and latent codes.
        Must contain: {RUN_NAME}/model/3000.pth, {RUN_NAME}/latent_codes/3000.pth,
        and {RUN_NAME}/model_params_config.json. Default: run_v57a

Output:
    {RUN_NAME}/latent_codes_average/
        family/
            mean_latents_by_family.npz   - NPZ with arrays 'family' (names) and 'latents'
            mean_latents_by_family.csv   - CSV with one row per family and latent dims as columns
            log_members_by_family.csv    - CSV mapping each VTK file to its family group
        genus/
            mean_latents_by_genus.{npz,csv}
            log_members_by_genus.csv
        species/
            mean_latents_by_species.{npz,csv}
            log_members_by_species.csv

Notes:
    Taxonomic names are parsed from VTK filenames via vtk_parsing_logic.parse_vtk_filename.
    Meshes with unparseable filenames at a given level are excluded from that level's grouping.
    Checkpoint is hardcoded to epoch 3000 (CKPT = '3000').
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from NSM.helper_funcs import load_config, load_model_and_latents
from vtk_parsing_logic import parse_vtk_filename

# ============================================================================
# Configuration
# ============================================================================
_parser = argparse.ArgumentParser()
_parser.add_argument('--run_name', type=str, default='run_v57a', help='Training directory name (default: run_v57a)')
_args = _parser.parse_args()

TRAIN_DIR = _args.run_name
CKPT = '3000'
LC_PATH = f'{TRAIN_DIR}/latent_codes/{CKPT}.pth'
MODEL_PATH = f'{TRAIN_DIR}/model/{CKPT}.pth'
# Default config path is TRAIN_DIR/model_params_config.json, which should be saved after every model run
CONFIG_PATH = f'{TRAIN_DIR}/model_params_config.json'

# Output directories
OUTPUT_BASE = f'./{TRAIN_DIR}/latent_codes_average'
OUTPUT_FAMILY = os.path.join(OUTPUT_BASE, 'family')
OUTPUT_GENUS = os.path.join(OUTPUT_BASE, 'genus')
OUTPUT_SPECIES = os.path.join(OUTPUT_BASE, 'species')


# ============================================================================
# Functions
# ============================================================================

def compute_mean_latents_by_group(latent_codes, parsed_info_list, group_key):
    """
    Compute mean latent vectors grouped by a specific key (family, genus, species).

    Args:
        latent_codes: torch.Tensor of shape [N, latent_dim]
        parsed_info_list: list of parsed filename dicts
        group_key: str, one of 'family', 'genus', 'species'

    Returns:
        tuple: (mean_latents dict, groups dict)
        - mean_latents: dict mapping group_name -> mean latent vector (numpy array)
        - groups: dict mapping group_name -> list of indices
    """
    # Group indices by the group key
    groups = defaultdict(list)

    for idx, info in enumerate(parsed_info_list):
        group_name = info.get(group_key)
        if group_name:
            groups[group_name].append(idx)

    # Compute mean for each group
    latent_np = latent_codes.cpu().numpy()
    mean_latents = {}

    for group_name, indices in tqdm(groups.items(), desc=f"Computing {group_key} means"):
        group_latents = latent_np[indices]
        mean_latents[group_name] = group_latents.mean(axis=0)

    return mean_latents, dict(groups)


def save_latents(mean_latents, output_dir, group_key):
    """
    Save mean latent vectors to NPZ and CSV files.

    Args:
        mean_latents: dict mapping group_name -> mean latent vector
        output_dir: str, output directory path
        group_key: str, name of grouping level (for column naming)
    """
    os.makedirs(output_dir, exist_ok=True)

    if not mean_latents:
        print(f"Warning: No data to save for {group_key}")
        return

    # Determine latent dimension from first entry
    first_latent = next(iter(mean_latents.values()))
    latent_dim = len(first_latent)

    # Prepare data for DataFrame
    data = []
    for group_name, latent_vec in sorted(mean_latents.items()):
        row = {group_key: group_name}
        for i in range(latent_dim):
            row[f'lt_{i+1:04d}'] = latent_vec[i]
        data.append(row)

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = os.path.join(output_dir, f'mean_latents_by_{group_key}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save as NPZ
    npz_path = os.path.join(output_dir, f'mean_latents_by_{group_key}.npz')
    # Create arrays for NPZ
    group_names = np.array(sorted(mean_latents.keys()))
    latent_array = np.array([mean_latents[name] for name in group_names])
    np.savez(npz_path, **{group_key: group_names, 'latents': latent_array})
    print(f"Saved NPZ: {npz_path}")

    return df


def save_group_logs(groups, parsed_info_list, list_mesh_paths, output_dir, group_key):
    """
    Save log of which VTK files and latent indices were used for each group.

    Args:
        groups: dict mapping group_name -> list of indices
        parsed_info_list: list of parsed filename dicts
        list_mesh_paths: list of original mesh paths
        output_dir: str, output directory path
        group_key: str, name of grouping level
    """
    os.makedirs(output_dir, exist_ok=True)

    if not groups:
        print(f"Warning: No groups to log for {group_key}")
        return

    # Create log data for all entries
    log_data = []
    for group_name in sorted(groups.keys()):
        indices = groups[group_name]
        for idx in indices:
            info = parsed_info_list[idx]
            vtk_path = list_mesh_paths[idx]
            vtk_filename = os.path.basename(vtk_path)

            log_data.append({
                'vtk_filename': vtk_filename,
                'latent_index': idx,
                'family': info.get('family', ''),
                'genus': info.get('genus', ''),
                'species': info.get('species', ''),
            })

    df_log = pd.DataFrame(log_data)

    # Sort by the group key, then by latent index
    df_log = df_log.sort_values(by=[group_key, 'latent_index']).reset_index(drop=True)

    # Save log CSV
    log_path = os.path.join(output_dir, f'log_members_by_{group_key}.csv')
    df_log.to_csv(log_path, index=False)
    print(f"Saved log: {log_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Create Mean Latent Vectors per Taxonomic Group")
    print("=" * 60)

    # Load config
    print(f"\nLoading config from: {CONFIG_PATH}")
    config = load_config(config_path=CONFIG_PATH)
    device = config.get("device", "cuda:0")

    # Get list of mesh paths
    list_mesh_paths = config['list_mesh_paths']
    print(f"Found {len(list_mesh_paths)} mesh paths in config")

    # Load model and latent codes
    print(f"\nLoading model from: {MODEL_PATH}")
    print(f"Loading latent codes from: {LC_PATH}")
    model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

    # Verify dimensions match
    num_meshes = len(list_mesh_paths)
    num_latents = latent_codes.shape[0]
    latent_dim = latent_codes.shape[1]

    print(f"\nNumber of mesh paths: {num_meshes}")
    print(f"Number of latent codes: {num_latents}")
    print(f"Latent dimension: {latent_dim}")

    assert num_meshes == num_latents, \
        f"Mismatch! Number of mesh paths ({num_meshes}) != number of latent codes ({num_latents})"
    print("✓ Dimension check passed: mesh paths and latent codes match")

    # Parse all VTK filenames
    print("\nParsing VTK filenames...")
    parsed_info_list = []
    for path in tqdm(list_mesh_paths, desc="Parsing filenames"):
        parsed_info_list.append(parse_vtk_filename(path))

    # Print some statistics
    families = set(info['family'] for info in parsed_info_list if info['family'])
    genera = set(info['genus'] for info in parsed_info_list if info['genus'])
    species = set(info['species'] for info in parsed_info_list if info['species'])

    print(f"\nUnique families: {len(families)}")
    print(f"Unique genera: {len(genera)}")
    print(f"Unique species: {len(species)}")

    # Create output directories
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(OUTPUT_FAMILY, exist_ok=True)
    os.makedirs(OUTPUT_GENUS, exist_ok=True)
    os.makedirs(OUTPUT_SPECIES, exist_ok=True)

    # Compute and save mean latents for each taxonomic level
    print("\n" + "=" * 60)
    print("Computing mean latent vectors by FAMILY")
    print("=" * 60)
    mean_latents_family, groups_family = compute_mean_latents_by_group(latent_codes, parsed_info_list, 'family')
    save_latents(mean_latents_family, OUTPUT_FAMILY, 'family')
    save_group_logs(groups_family, parsed_info_list, list_mesh_paths, OUTPUT_FAMILY, 'family')

    print("\n" + "=" * 60)
    print("Computing mean latent vectors by GENUS")
    print("=" * 60)
    mean_latents_genus, groups_genus = compute_mean_latents_by_group(latent_codes, parsed_info_list, 'genus')
    save_latents(mean_latents_genus, OUTPUT_GENUS, 'genus')
    save_group_logs(groups_genus, parsed_info_list, list_mesh_paths, OUTPUT_GENUS, 'genus')

    print("\n" + "=" * 60)
    print("Computing mean latent vectors by SPECIES")
    print("=" * 60)
    mean_latents_species, groups_species = compute_mean_latents_by_group(latent_codes, parsed_info_list, 'species')
    save_latents(mean_latents_species, OUTPUT_SPECIES, 'species')
    save_group_logs(groups_species, parsed_info_list, list_mesh_paths, OUTPUT_SPECIES, 'species')

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_BASE}")
    print(f"  - Family: {OUTPUT_FAMILY}")
    print(f"  - Genus: {OUTPUT_GENUS}")
    print(f"  - Species: {OUTPUT_SPECIES}")


if __name__ == "__main__":
    main()
