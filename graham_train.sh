#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:t4:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=55GB
#SBATCH --account=def-ester
#SBATCH --output=graham_logp_50_1_cont.out
echo “starting the job...”
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.
module load python/3.8
source ~/scratch/env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json