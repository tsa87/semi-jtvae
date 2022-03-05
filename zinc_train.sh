#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --account=rrg-ester
#SBATCH --output=out.out
echo “starting the job...”
export PYTHONPATH=.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source ~/.bashrc
module load python/3.8
source anb_env/bin/activate
python training/run_experiment.py --config_path=training/configs/rand_gen_zinc250k_config_dict.json
