#!/usr/bin/env bash
#SBATCH --job-name=mer-tune
#SBATCH --array=0-99%12     # create 100 tasks 0-99, each
#SBATCH --cpus-per-task=1   # * running on 1 CPU core
#SBATCH --mem=2G            # * reserving 2 GB RAM
#SBATCH --time=01:00:00     # * having a timeout of one hour (usually takes much less)

# tasks will not be further parallelized
export OPT_NUM_PROC=1
export OPT_NUM_TRIALS=1

# read parameters from environment variables
STEPS="${STEPS:-1000000}"                 # number of steps per trial
NORM_BASE="${NORM_BASE:-norms}"           # norm base to use and monitor
MIN_WEIGHT="${MIN_WEIGHT:-10}"            # minimum weight (punishment) to be assigned to any norm violation
MAX_WEIGHT="${MAX_WEIGHT:-1000}"          # maximum weight (punishment) to be assigned to any norm violation
export NORMS_ORDERING_BASE="${BASE:-10}"  # base for weighing the different norms against each other

# the optuna study title
export STUDY_NAME=merchant_${NORM_BASE}_b${NORMS_ORDERING_BASE}_${MIN_WEIGHT}-${MAX_WEIGHT}_${STEPS}

# stagger trial according to task id
if [ $SLURM_ARRAY_TASK_ID -lt 12 ]; then sleep $(( 10 * SLURM_ARRAY_TASK_ID )); fi

# run a single trial for the given task
python ../train_hyper.py --env merchant --algo Q --norm-base ${NORM_BASE} --norms-tune ${MIN_WEIGHT} ${MAX_WEIGHT} --seeds 5 --steps ${STEPS}
