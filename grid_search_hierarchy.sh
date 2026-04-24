#!/bin/bash
# ==============================================================================
# grid_search_hierarchy.sh
#
# Submit a grid-search sweep over hierarchy-loss hyperparameters. Each grid
# point becomes its own sbatch submission using train_classify.slurm.
#
# Usage (on HPC login node):
#   cd /blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM
#   bash grid_search_hierarchy.sh
#
# Default grid: 3 values of --contrastive-weight x 3 values of --head-weight
#               = 9 sbatch submissions.
#
# To extend the sweep, edit the arrays below. Full Cartesian product is
# submitted, so adding dimensions multiplies job count quickly
# (e.g. adding a 3-value --species-weight sweep -> 27 jobs).
#
# Each job:
#   - Gets a unique --run-name reflecting its hyperparams, so output dirs are
#     self-labelling (e.g. hierarchy_grid_cw0.01_hw0.005/).
#   - Writes {run_dir}/hyperparams.json with the exact weights used.
#   - Appends a row to the manifest CSV emitted by this script.
# ==============================================================================

set -euo pipefail

# ---------- Grid definition ---------------------------------------------------
CONTRASTIVE_WEIGHTS=(0.001 0.01 0.1)
HEAD_WEIGHTS=(0.0005 0.005 0.05)

# To sweep additional parameters, uncomment and add nested loops below:
# SPECIES_WEIGHTS=(0.5 1.0 2.0)
# GENUS_WEIGHTS=(0.25 0.5 1.0)
# FAMILY_WEIGHTS=(0.1 0.25 0.5)
# POSITION_WEIGHTS=(0.5 0.75 1.0)

# ---------- Sanity checks -----------------------------------------------------
if [[ ! -f train_classify.slurm ]]; then
    echo "ERROR: train_classify.slurm not found in $(pwd)."
    echo "Run this script from the project root on HPC."
    exit 1
fi

if [[ ! -f run_train_hierarchy.py ]]; then
    echo "ERROR: run_train_hierarchy.py not found in $(pwd)."
    exit 1
fi

# ---------- Manifest header ---------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="grid_manifest_${TIMESTAMP}.csv"
echo "job_id,run_name,contrastive_weight,head_weight,submitted_at" > "$MANIFEST"

echo "================================================================"
echo "Grid search sweep"
echo "  contrastive weights: ${CONTRASTIVE_WEIGHTS[*]}"
echo "  head weights:        ${HEAD_WEIGHTS[*]}"
echo "  total jobs:          $(( ${#CONTRASTIVE_WEIGHTS[@]} * ${#HEAD_WEIGHTS[@]} ))"
echo "  manifest:            $MANIFEST"
echo "================================================================"

# ---------- Submit ------------------------------------------------------------
SUBMITTED=0
SKIPPED=0

for cw in "${CONTRASTIVE_WEIGHTS[@]}"; do
    for hw in "${HEAD_WEIGHTS[@]}"; do
        RUN_NAME="hierarchy_grid_cw${cw}_hw${hw}"

        if [[ -d "$RUN_NAME" ]]; then
            echo "SKIP: $RUN_NAME already exists (delete it to re-run this cell)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        OUT=$(sbatch \
            --job-name="gridhx_cw${cw}_hw${hw}" \
            train_classify.slurm \
            --run-name "$RUN_NAME" \
            --contrastive-weight "$cw" \
            --head-weight "$hw")

        # sbatch prints: "Submitted batch job <id>"
        JOB_ID=$(echo "$OUT" | awk '{print $NF}')
        echo "SUBMIT: job=$JOB_ID  run=$RUN_NAME  cw=$cw  hw=$hw"
        echo "$JOB_ID,$RUN_NAME,$cw,$hw,$(date -Iseconds)" >> "$MANIFEST"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "================================================================"
echo "Done. Submitted $SUBMITTED jobs, skipped $SKIPPED."
echo "Manifest: $MANIFEST"
echo "Watch queue: squeue -u \$USER"
echo "Cancel all: awk -F, 'NR>1 {print \$1}' $MANIFEST | xargs scancel"
echo "================================================================"
