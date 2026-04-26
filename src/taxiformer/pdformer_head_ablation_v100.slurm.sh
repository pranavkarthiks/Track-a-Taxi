#!/bin/bash
#SBATCH --job-name=pdformer-head-ablation
#SBATCH --account=coms038604
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=pdformer-head-ablation-%j.out
#SBATCH --error=pdformer-head-ablation-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

bash run_pdformer_head_ablation_base.sh
