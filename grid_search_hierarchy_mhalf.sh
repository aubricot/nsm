#!/bin/bash
# ==============================================================================
# grid_search_hierarchy_mhalf.sh
#
# Third sister script (alongside grid_search_hierarchy.sh and
# grid_search_hierarchy_m0248.sh). Sweeps the same (cw × hw) grid but with a
# *conservative* taxonomic-margin table — half the default values:
#
#   margins = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}     ← this script  (mhalf)
#
# vs. the two existing sweeps:
#
#   margins = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}     ← grid_search_hierarchy.sh
#   margins = {0: 0.0, 1: 2.0, 2: 4.0, 3: 8.0}     ← grid_search_hierarchy_m0248.sh
#
# Why these values: latents are L2-normalized in HierarchyContrastiveLoss, so
# the maximum achievable Euclidean distance between any two latents is 2.0.
# The mhalf table keeps every margin at-or-below that ceiling, so the loss
# can actually settle at zero for well-separated negative pairs instead of
# being permanently saturated. Tests whether *headroom* in the loss surface
# helps optimization vs the always-on push of the default and m0248 tables.
#
# Usage (on HPC login node):
#   cd /blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM
#   bash grid_search_hierarchy_mhalf.sh
# ==============================================================================

set -euo pipefail

# ---------- Grid definition (same as the other two sweeps) -------------------
CONTRASTIVE_WEIGHTS=(0.001 0.01 0.1)
HEAD_WEIGHTS=(0.0005 0.005 0.05)

# ---------- Margin table (the only thing that differs) ----------------------
MARGIN_TAG="mhalf"
M0=0.0
M1=0.5
M2=1.0
M3=2.0

# ---------- Sanity checks ----------------------------------------------------
if [[ ! -f train_classify.slurm ]]; then
    echo "ERROR: train_classify.slurm not found in $(pwd)."
    echo "Run this script from the project root on HPC."
    exit 1
fi

if [[ ! -f run_train_hierarchy.py ]]; then
    echo "ERROR: run_train_hierarchy.py not found in $(pwd)."
    exit 1
fi

# ---------- Sweep identifier -------------------------------------------------
SWEEP_TS=$(date +%Y%m%d_%H%M%S)
MANIFEST="grid_manifest_${MARGIN_TAG}_${SWEEP_TS}.csv"
echo "job_id,run_name,contrastive_weight,head_weight,margin_set,sweep_ts,submitted_at" > "$MANIFEST"

echo "================================================================"
echo "Grid search sweep ${SWEEP_TS}  [margins=$MARGIN_TAG → $M0 $M1 $M2 $M3]"
echo "  contrastive weights: ${CONTRASTIVE_WEIGHTS[*]}"
echo "  head weights:        ${HEAD_WEIGHTS[*]}"
echo "  total jobs:          $(( ${#CONTRASTIVE_WEIGHTS[@]} * ${#HEAD_WEIGHTS[@]} ))"
echo "  manifest:            $MANIFEST"
echo "  run-dir prefix:      hierarchy_grid_${SWEEP_TS}_..._${MARGIN_TAG}"
echo "================================================================"

# ---------- Submit -----------------------------------------------------------
SUBMITTED=0
SKIPPED=0

for cw in "${CONTRASTIVE_WEIGHTS[@]}"; do
    for hw in "${HEAD_WEIGHTS[@]}"; do
        RUN_NAME="hierarchy_grid_${SWEEP_TS}_cw${cw}_hw${hw}_${MARGIN_TAG}"

        if [[ -d "$RUN_NAME" ]]; then
            echo "SKIP: $RUN_NAME already exists (delete it to re-run this cell)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        OUT=$(sbatch \
            --job-name="gridhx_${MARGIN_TAG}_${SWEEP_TS}_cw${cw}_hw${hw}" \
            train_classify.slurm \
            --run-name "$RUN_NAME" \
            --contrastive-weight "$cw" \
            --head-weight "$hw" \
            --contrastive-margins "$M0" "$M1" "$M2" "$M3")

        JOB_ID=$(echo "$OUT" | awk '{print $NF}')
        echo "SUBMIT: job=$JOB_ID  run=$RUN_NAME  cw=$cw  hw=$hw  margins=$M0,$M1,$M2,$M3"
        echo "$JOB_ID,$RUN_NAME,$cw,$hw,$MARGIN_TAG,$SWEEP_TS,$(date -Iseconds)" >> "$MANIFEST"
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
echo ""
echo "Three-way comparison workflow:"
echo "  After all three sweeps complete, run:"
echo "    python run_grid_report.py"
echo "  The report's heatmaps/ dir will contain three groups per metric:"
echo "    *_default.png   (margins 0,1,2,4)"
echo "    *_m0248.png     (margins 0,2,4,8)"
echo "    *_mhalf.png     (margins 0,0.5,1,2)"
echo "  master_grid_comparison.csv records the full margin table per row in"
echo "  the contrastive_margins column for direct numeric comparison."
