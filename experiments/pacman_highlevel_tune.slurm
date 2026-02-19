#!/usr/bin/env bash
#SBATCH --job-name=pmh-tune
#SBATCH --array=0-99%16     # create 100 tasks 0-99, each
#SBATCH --cpus-per-task=1   # * running on 1 CPU core
#SBATCH --mem=5G            # * reserving 5 GB RAM
#SBATCH --time=167:59:59    # * having a timeout of seven days (usually takes much less)

# tasks will not be further parallelized
export OPT_NUM_PROC=1
export OPT_NUM_TRIALS=1

# read parameters from environment variables
ALGO="${ALGO:-PPO}"                       # the algoright (PPO or DQN)
STEPS="${STEPS:-2500000}"                 # number of steps per trial
NORM_BASE="${NORM_BASE:-vegan}"           # norm base to use and monitor
MIN_WEIGHT="${MIN_WEIGHT:-50}"            # minimum weight (punishment) to be assigned to any norm violation
MAX_WEIGHT="${MAX_WEIGHT:-5000}"          # maximum weight (punishment) to be assigned to any norm violation
export NORMS_ORDERING_BASE="${BASE:-10}"  # base for weighing the different norms against each other

# the optuna study title
export STUDY_NAME=pm_highlevel_tune_${NORM_BASE}_${ALGO}_b${NORMS_ORDERING_BASE}_${MIN_WEIGHT}-${MAX_WEIGHT}_${STEPS}

# stagger trial according to task id
if [ $SLURM_ARRAY_TASK_ID -lt 16 ]; then sleep $(( 10 * SLURM_ARRAY_TASK_ID )); fi

# run a single trial for the given task
python ../train_hyper.py --algo ${ALGO} --norm-base ${NORM_BASE} --norms-tune ${MIN_WEIGHT} ${MAX_WEIGHT} --seeds 5 --steps ${STEPS}
