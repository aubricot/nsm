#!/bin/bash
#SBATCH --job-name=classification_nsm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mdelage6@gatech.edu
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM/logs/%x_%j.log
#SBATCH --error=/blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="/blue/arthur.porto-biocosmos/mdelage6.gatech/classification-NSM"
CONDA_BASE="/apps/conda/25.7.0"
CONDA_ENV="/blue/arthur.porto-biocosmos/mdelage6.gatech/conda_envs/NSM"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

on_error() {
    local exit_code=$?
    echo "ERROR: Script failed with exit code $exit_code at line ${BASH_LINENO[0]}"
    date
    exit "$exit_code"
}
trap on_error ERR

echo "================ JOB INFO ================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node List: ${SLURM_JOB_NODELIST:-N/A}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR:-N/A}"
echo "Working Dir: $(pwd)"
echo "Hostname: $(hostname)"
date
echo "=========================================="

echo "Changing to project directory..."
cd "$PROJECT_DIR"

echo "Loading conda..."
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Activating conda environment..."
conda activate "$CONDA_ENV"

echo "Environment checks..."
echo "Python: $(which python)"
python --version
echo "Conda env: $CONDA_DEFAULT_ENV"

echo "GPU status..."
nvidia-smi || echo "WARNING: nvidia-smi failed"

echo "SLURM GPU info..."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "Starting training... (extra args: $*)"
python run_train_hierarchy.py "$@"

echo "Training python exited cleanly."
date

# ---- Resolve run directory written by train.py ------------------------------
SENTINEL=".last_train_run.${SLURM_JOB_ID:-}"
[[ -f "$SENTINEL" ]] || SENTINEL=".last_train_run"
if [[ ! -f "$SENTINEL" ]]; then
    echo "ERROR: no .last_train_run sentinel was written. Training likely never started."
    exit 1
fi
RUN_DIR=$(cat "$SENTINEL")

# ---- Post-training sanity checks --------------------------------------------
# Training is "successful" only if the expected artifacts exist. Otherwise the
# python process may have exited 0 without doing real work (e.g. silent dead-
# code path, dataset loading short-circuit, OOM that aborted between epochs).
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory '$RUN_DIR' does not exist after training."
    echo "       Training python exited 0 but produced no output directory."
    exit 1
fi
if [[ ! -d "$RUN_DIR/model" ]] || ! ls "$RUN_DIR/model"/*.pth >/dev/null 2>&1; then
    echo "ERROR: no model checkpoints in $RUN_DIR/model/. Training did not reach a save_frequency epoch."
    ls -la "$RUN_DIR" || true
    exit 1
fi
if [[ ! -d "$RUN_DIR/latent_codes" ]] || ! ls "$RUN_DIR/latent_codes"/*.pth >/dev/null 2>&1; then
    echo "ERROR: no latent codes in $RUN_DIR/latent_codes/. Training output is incomplete."
    ls -la "$RUN_DIR" || true
    exit 1
fi
echo "Sanity check passed: $RUN_DIR has model+latent checkpoints."
echo "  model:        $(ls "$RUN_DIR/model"/*.pth | wc -l) file(s)"
echo "  latent_codes: $(ls "$RUN_DIR/latent_codes"/*.pth | wc -l) file(s)"
echo "  hyperparams.json: $([[ -f "$RUN_DIR/hyperparams.json" ]] && echo yes || echo MISSING)"
echo "  config.json:      $([[ -f "$RUN_DIR/config.json" ]] && echo yes || echo MISSING)"

echo "Starting classification on run directory: $RUN_DIR"
python run_classify.py --run-dir "$RUN_DIR"

echo "Classification finished successfully."
date

echo "Job completed successfully."