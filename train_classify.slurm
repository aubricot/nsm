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

echo "Starting training..."
python train_model_hierarchy.py

echo "Training finished successfully."
date

echo "Starting classification..."
python classify_vertebrae_v5.py

echo "Classification finished successfully."
date

echo "Job completed successfully."