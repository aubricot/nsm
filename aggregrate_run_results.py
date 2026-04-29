"""
Aggregate summary_matches CSV files from classify_vertebrae_normpos.py into a
single combined results CSV with per-run statistics.

The results from classify_vertebrae_normpos.py are saved in my_run/classify_vertebrae/results
However, these need to be aggregated into a single CSV for easier analysis and comparison across runs.

Usage:
    python aggregrate_run_results.py [input_dir]
    python aggregrate_run_results.py --subfolder my_run/classify_vertebrae/results
    python aggregrate_run_results.py --subfolder C:/full/path/to/results

Arguments:
    input_dir           Full directory containing summary_matches*.csv files (default: current directory)
    --subfolder         Relative or absolute path to search for CSVs (overrides input_dir)

Output:
    {input_dir or subfolder}/combined_results/combined_results.csv
        One row per input CSV with columns including:
        - run, version, noise, init_method, init_success_true_count
        - species_matched, species_matched_pct
        - top1_region_pct, top1_pos_err_avg_int_only, top1_pos_err_avg_with_penalty
        - top5_species_match_pct, top5_region_pct, top5_pos_err_avg_with_penalty
        - similar_1_distance_{min,mean,max}

Notes:
    Position error penalty: rows where species did not match are penalized with a
    value of 1 when computing top1/top5_pos_err_avg_with_penalty.
    Run name and noise level are auto-extracted from filenames via regex
    (e.g. genus_v57aaaa_sn0.3 -> version=v57aaaa, noise=0.3).
"""

import pandas as pd
import os
import glob
import argparse
import re


def detect_prefixes(filenames):
    """Auto-detect date prefixes like summary_matches_2026-02-03_ from filenames."""
    prefixes = set()
    for f in filenames:
        match = re.match(r"(summary_matches_\d{4}-\d{2}-\d{2}_)", os.path.basename(f))
        if match:
            prefixes.add(match.group(1))
    return sorted(prefixes)


def compute_position_error_stats(series, species_match_series):
    """Compute stats for a position_error column.

    Penalty logic: rows where species_match != 'yes' are penalized +1
    (replacing their position_error value with 1 in the penalized average).
    """
    na_mismatch_count = int((series.astype(str).str.strip() == "NA_region_mismatch").sum())
    numeric = pd.to_numeric(series, errors="coerce")
    int_values = numeric.dropna()
    avg_int_only = round(int_values.mean(), 4) if len(int_values) > 0 else None

    # Penalize rows where species did not match (replace their value with 1)
    species_mismatch = species_match_series.str.lower() != "yes"
    combined = []
    for pos_err, mismatch in zip(pd.to_numeric(series, errors="coerce"), species_mismatch):
        if mismatch:
            combined.append(1)
        elif pd.notna(pos_err):
            combined.append(pos_err)
    avg_with_penalty = round(sum(combined) / len(combined), 4) if len(combined) > 0 else None
    return na_mismatch_count, avg_int_only, avg_with_penalty


def process_csv(filepath, prefixes):
    """Process a single summary_matches CSV and return a summary row."""
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath)

    # Strip prefixes and extension to get run name
    run = filename
    for prefix in prefixes:
        run = run.replace(prefix, "")
    run = run.replace(".csv", "")

    total = len(df)
    init_method = df["init_method"].iloc[0]
    if "init_success" in df.columns:
        init_success_count = int((df["init_success"] == True).sum())
    else:
        init_success_count = total  # cascade format has no init_success; treat all as True

    # Top-1 species match
    species_matched = int((df["top1_species_match"].str.lower() == "yes").sum())
    species_not_matched = total - species_matched
    species_pct = (species_matched / total) * 100 if total > 0 else 0

    # Top-1 region match
    top1_region_yes = int((df["top1_region_match"].str.lower() == "yes").sum())
    top1_region_no = total - top1_region_yes
    top1_region_pct = (top1_region_yes / total) * 100 if total > 0 else 0

    # Top-1 position error
    top1_na_count, top1_avg_int, top1_avg_penalty = compute_position_error_stats(df["top1_position_error"], df["top1_species_match"])

    # Top-5 species match
    top5_matched = int((df["top5_species_match"].str.lower() == "yes").sum())
    top5_not_matched = total - top5_matched
    top5_pct = (top5_matched / total) * 100 if total > 0 else 0

    # Top-5 region match
    top5_region_yes = int((df["top5_region_match"].str.lower() == "yes").sum())
    top5_region_no = total - top5_region_yes
    top5_region_pct = (top5_region_yes / total) * 100 if total > 0 else 0

    # Top-5 position error
    top5_na_count, top5_avg_int, top5_avg_penalty = compute_position_error_stats(df["top5_position_error"], df["top5_species_match"])

    min_dist = df["similar_1_distance"].min()
    mean_dist = df["similar_1_distance"].mean()
    max_dist = df["similar_1_distance"].max()

    # Extract version from run name (e.g. genus_v57aaaa -> v57aaaa)
    version_match = re.search(r"_(v[A-Za-z0-9]+)", run)
    version = version_match.group(1) if version_match else ""

    # Extract noise level from run name (e.g. genus_v57aaaa_sn0.3 -> 0.3)
    noise_match = re.search(r"_sn(\d+(?:\.\d+)?)", run)
    noise = float(noise_match.group(1)) if noise_match else 0.0

    return {
        "run": run,
        "version": version,
        "noise": noise,
        "init_method": init_method,
        "init_success_true_count": init_success_count,
        "species_matched": species_matched,
        "species_not_matched": species_not_matched,
        "species_matched_pct": round(species_pct, 2),
        "top1_region_yes": top1_region_yes,
        "top1_region_no": top1_region_no,
        "top1_region_pct": round(top1_region_pct, 2),
        "top1_pos_err_na_mismatch_count": top1_na_count,
        "top1_pos_err_avg_int_only": top1_avg_int,
        "top1_pos_err_avg_with_penalty": top1_avg_penalty,
        "top5_species_match": top5_matched,
        "top5_species_not_match": top5_not_matched,
        "top5_species_match_pct": round(top5_pct, 2),
        "top5_region_yes": top5_region_yes,
        "top5_region_no": top5_region_no,
        "top5_region_pct": round(top5_region_pct, 2),
        "top5_pos_err_na_mismatch_count": top5_na_count,
        "top5_pos_err_avg_int_only": top5_avg_int,
        "top5_pos_err_avg_with_penalty": top5_avg_penalty,
        "similar_1_distance_min": round(min_dist, 6),
        "similar_1_distance_mean": round(mean_dist, 6),
        "similar_1_distance_max": round(max_dist, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Combine summary_matches CSVs into a single results summary.")
    parser.add_argument("input_dir", nargs="?", default=".", help="Directory containing summary_matches*.csv files")
    parser.add_argument("--subfolder", default=None, help="Subfolder name, relative path, or full path to search inside (overrides input_dir)")
    args = parser.parse_args()

    if args.subfolder is not None:
        search_dir = os.path.abspath(args.subfolder)
    else:
        search_dir = os.path.abspath(args.input_dir)

    csv_files = sorted(glob.glob(os.path.join(search_dir, "summary_matches*.csv")))
    if not csv_files:
        print(f"No summary_matches*.csv files found in {search_dir}")
        return

    prefixes = detect_prefixes(csv_files)
    print(f"Found {len(csv_files)} files")
    print(f"Stripping prefixes: {prefixes}")

    rows = [process_csv(f, prefixes) for f in csv_files]
    result_df = pd.DataFrame(rows)

    out_dir = os.path.join(search_dir, "combined_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combined_results.csv")
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(result_df)} rows to {out_path}")
    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
