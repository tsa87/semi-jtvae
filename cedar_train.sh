#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=12GB
#SBATCH --account=def-ester
#SBATCH --output=cedar_logp_50_1_0.out
echo “starting the job...”
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.
module load python/3.8
source ~/scratch/env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json
