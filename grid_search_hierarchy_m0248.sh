#!/bin/bash
# ==============================================================================
# grid_search_hierarchy_m0248.sh
#
# Sister script to grid_search_hierarchy.sh that sweeps the same (cw × hw)
# grid but with a *different* taxonomic-margin table:
#
#   margins = {0: 0.0, 1: 2.0, 2: 4.0, 3: 8.0}     ← this script
#
# vs. the default used by grid_search_hierarchy.sh:
#
#   margins = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}     ← that script
#
# Same 9 sbatch submissions per sweep, identical infrastructure (sentinel
# handoff, post-training sanity checks, manifest CSV). Run-name and sbatch
# job-name include `_m0248` so the new runs are visually distinct from the
# previous sweep's `_m0124` runs in `ls`, `squeue`, and grid_status.sh.
#
# Usage (on HPC login node):
#   cd /blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM
#   bash grid_search_hierarchy_m0248.sh
#
# Caveat — latents are L2-normalized in HierarchyContrastiveLoss, so the
# maximum achievable Euclidean distance between any two latents is 2.0.
# Margins of 4.0 and 8.0 are therefore unreachable; their `(margin - d)^2`
# penalty is permanently active for every pair of those taxonomic distances.
# This sweep tests whether that "always-on push" helps separation, or
# whether it saturates the gradient signal.
# ==============================================================================

set -euo pipefail

# ---------- Grid definition (same as the default-margin sweep) ---------------
CONTRASTIVE_WEIGHTS=(0.001 0.01 0.1)
HEAD_WEIGHTS=(0.0005 0.005 0.05)

# ---------- Margin table (the only thing that differs from the sister) ------
MARGIN_TAG="m0248"
M0=0.0
M1=2.0
M2=4.0
M3=8.0

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
echo "Comparison workflow:"
echo "  After this sweep finishes, run:"
echo "    python run_grid_report.py"
echo "  The master_grid_comparison.csv will include both margin sets;"
echo "  filter on hierarchy_contrastive_margins (or run dir suffix) to"
echo "  compare $MARGIN_TAG against the default-margin sweep."
