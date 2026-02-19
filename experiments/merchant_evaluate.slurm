#!/usr/bin/env bash
#SBATCH --job-name=mer-eval
#SBATCH --array=0-19%8      # create 20 tasks 0-19, each
#SBATCH --cpus-per-task=1   # * running on 1 CPU core
#SBATCH --mem=2G            # * reserving 2 GB RAM
#SBATCH --time=01:00:00     # * having a timeout of one hour (usually takes much less)

# tasks will not be further parallelized
export OPT_NUM_PROC=1
export OPT_NUM_TRIALS=1

# read parameters from environment variables
STEPS="${STEPS:-1000000}"        # number of steps per trial
NORM_BASE="${NORM_BASE:-norms}"  # norm base to use and monitor

export STUDY_NAME=merchant_${NORM_BASE}_${STEPS}

if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then sleep $(( 10 * SLURM_ARRAY_TASK_ID )); fi

python ../train_hyper.py --env merchant --algo Q --norm-base ${NORM_BASE} --seeds 1 --steps ${STEPS} --save-models --log-csv
