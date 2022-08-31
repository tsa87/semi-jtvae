#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:v100l:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=15GB
#SBATCH --account=def-ester
#SBATCH --output=cedar0.out
echo “starting the job...”
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.
module load python/3.8
source ~/scratch/env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json