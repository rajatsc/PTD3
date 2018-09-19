#!/bin/bash

##-----Resource Request--------

#SBATCH --verbose
#SBATCH --job-name=td3
#SBATCH --output=td3_%j.out
#SBATCH --error=td3_%j.err
#SBATCH --workdir='/gscratch/stf/rajatc/TD3/env/halfcheetah/batch'
#SBATCH --account=stf
#SBATCH --partition=stf-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rajatc@uw.edu

#SBATCH --nodes=1
#SBATCH --gres=gpu:P100:1
#SBATCH --mem=100G
#SBATCH --time=3:00:00


##--------------Job Steps------------------

module purge
module load cuda/9.1.85.3
source activate td3_cu91

nvidia-smi
cd /gscratch/stf/rajatc/TD3/env/halfcheetah/batch

bash ./run_batch.sh
