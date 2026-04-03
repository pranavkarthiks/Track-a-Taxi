# Base model with base dataset

#!/bin/bash
#SBATCH --job-name=NYCTaxi-Base-PD
#SBATCH --account=coms038604
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100
#SBATCH --mem=20GB
#SBATCH --time=7-00:00:00
#SBATCH --output=log-job.out
#SBATCH --error=log-err.err



. ~/initMamba.sh
cd ./TaxiTransformer
conda activate pdformer
nvidia-smi
pip install -r requirements.txt
python run_model.py --task traffic_state_pred --model PDFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator
