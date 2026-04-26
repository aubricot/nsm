#!/bin/bash
# ==============================================================================
# grid_status.sh
#
# Print a one-line-per-run status table for every hierarchy_v* / hierarchy_grid_*
# directory in the project. Inspects the filesystem (no SLURM queries) so it
# works on any host that can see $PROJECT_DIR.
#
# Usage (run from project root):
#   bash grid_status.sh                       # scan all hierarchy_*/ dirs
#   bash grid_status.sh grid_manifest_X.csv   # only runs listed in this manifest
#                                             # (useful for tracking one sweep)
#
# Status column meanings:
#   PENDING     directory does not exist yet (job queued, or never started)
#   EMPTY       directory exists but no checkpoints and no hyperparams.json
#   NO_CKPT     hyperparams.json present but training never reached a save epoch
#   TRAINED     model checkpoints present; classify has not run yet
#   COMPLETE    model checkpoints + classify results both present
# ==============================================================================

set -euo pipefail

MANIFEST_FILE="${1:-}"

# ---------- Auto-detect latest manifest if none given -----------------------
if [[ -z "$MANIFEST_FILE" ]]; then
    LATEST=$(ls -t grid_manifest_*.csv 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST" ]]; then
        echo "(no manifest specified; latest is $LATEST — pass it as an arg to filter)"
    fi
fi

# ---------- Per-run inspector -----------------------------------------------
inspect_run() {
    local run_name="$1"
    local cw="-" hw="-" status="?" n_epochs=0 last_epoch="-" n_results=0

    if [[ ! -d "$run_name" ]]; then
        printf "%-55s %-9s %-9s %-12s %-14s %-8s\n" \
            "$run_name" "$cw" "$hw" "PENDING" "-" "-"
        return
    fi

    if [[ -f "$run_name/hyperparams.json" ]]; then
        # One python invocation per run dir; 9-27 grid points → negligible.
        read -r cw hw < <(python3 - "$run_name/hyperparams.json" <<'PY' 2>/dev/null || echo "? ?"
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    print(d.get('hierarchy_contrastive_weight', '?'),
          d.get('classification_head_weight', '?'))
except Exception:
    print('?', '?')
PY
        )
    fi

    if [[ -d "$run_name/model" ]]; then
        n_epochs=$(ls "$run_name/model"/*.pth 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$n_epochs" -gt 0 ]]; then
            last_epoch=$(ls "$run_name/model"/*.pth 2>/dev/null \
                         | xargs -n1 basename \
                         | sed 's/\.pth$//' \
                         | sort -n | tail -1)
        fi
    fi

    if [[ -d "$run_name/results" ]]; then
        n_results=$(find "$run_name/results" -maxdepth 1 -type d -name 'ablation_*' 2>/dev/null \
                    | wc -l | tr -d ' ')
    fi

    # Priority: real artifacts trump manifest presence. A legacy run with
    # checkpoints+results but no hyperparams.json is still COMPLETE.
    if [[ "$n_epochs" -gt 0 && "$n_results" -gt 0 ]]; then
        status="COMPLETE"
    elif [[ "$n_epochs" -gt 0 ]]; then
        status="TRAINED"
    elif [[ -f "$run_name/hyperparams.json" ]]; then
        status="NO_CKPT"
    else
        status="EMPTY"
    fi

    printf "%-55s %-9s %-9s %-12s %-14s %-8s\n" \
        "$run_name" "$cw" "$hw" "$status" "$last_epoch ($n_epochs)" "$n_results"
}

# ---------- Header ----------------------------------------------------------
printf "%-55s %-9s %-9s %-12s %-14s %-8s\n" \
    "RUN_NAME" "CW" "HW" "STATUS" "LAST_EPOCH(N)" "RESULTS"
echo "----------------------------------------------------------------------------------------------------------"

# ---------- Source: manifest CSV or filesystem scan -------------------------
TOTAL=0
SOURCE=""
if [[ -n "$MANIFEST_FILE" && -f "$MANIFEST_FILE" ]]; then
    SOURCE="manifest $MANIFEST_FILE"
    # Manifest CSV columns: job_id,run_name,cw,hw,sweep_ts,submitted_at
    # (older manifests may omit sweep_ts; we only consume the first two cols.)
    while IFS=, read -r job_id run_name _rest; do
        [[ "$job_id" == "job_id" ]] && continue   # skip header row
        [[ -z "$run_name" ]] && continue
        inspect_run "$run_name"
        TOTAL=$((TOTAL + 1))
    done < "$MANIFEST_FILE"
else
    if ! compgen -G "hierarchy_*" >/dev/null; then
        echo "(no hierarchy_*/ directories found in $(pwd))"
        exit 0
    fi
    SOURCE="filesystem scan"
    for dir in hierarchy_*/; do
        [[ -d "$dir" ]] || continue
        inspect_run "${dir%/}"
        TOTAL=$((TOTAL + 1))
    done
fi

# ---------- Footer ----------------------------------------------------------
echo "----------------------------------------------------------------------------------------------------------"
echo "Source: $SOURCE  |  $TOTAL run(s) inspected"
echo ""
echo "Tips:"
echo "  - Live SLURM queue:           squeue -u \$USER"
echo "  - Tail a job log:             tail -f logs/<jobname>_<jobid>.log"
echo "  - Re-run only failed cells:   delete the failed run dirs, then bash grid_search_hierarchy.sh"
