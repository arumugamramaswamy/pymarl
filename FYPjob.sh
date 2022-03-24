#!/bin/sh

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=TestJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

EXP_NAME="test"

# module load anaconda
# source activate TestEnv
# python3 src/main.py --config=coma --env-config=test
python3 discord_webhook.py "Experiment name: $EXP_NAME"
