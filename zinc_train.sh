#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=45GB
#SBATCH --account=def-ester
#SBATCH --output=out.out
echo “starting the job...”
export PYTHONPATH=.
source ~/.bashrc
module load python/3.8
source env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json
