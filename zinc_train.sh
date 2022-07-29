#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=45GB
#SBATCH --account=def-ester
#SBATCH --output=out.out
echo “starting the job...”
echo $CUDA_VISIBLE_DEVICES
module load python/3.8
source ~/scratch/env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json
