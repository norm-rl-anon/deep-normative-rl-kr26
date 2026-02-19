#!/usr/bin/env bash
#SBATCH --job-name=pmi-eval
#SBATCH --array=0-0         # create a single task 0,
#SBATCH --cpus-per-task=8   # * running on 8 CPU cores
#SBATCH --gres=gpu:1        # * running on 1 GPU
#SBATCH --mem=200G          # * reserving 200 GB RAM
#SBATCH --time=167:59:59    # * having a timeout of seven days (usually takes much less)

# do 5 trials in parallel per task
export OPT_NUM_PROC=5
export OPT_NUM_TRIALS=5

# read parameters from environment variables
ALGO="${ALGO:-DQN}"              # the algoright (PPO or DQN)
STEPS="${STEPS:-20000000}"       # number of steps per trial
NORM_BASE="${NORM_BASE:-vegan}"  # norm base to use and monitor

if [ "$ALGO" = "DQN" ]; then
  hpfile="../hyperparams/images_default_dqn.json"
elif [ "$ALGO" = "PPO" ]; then
  hpfile="../hyperparams/images_default_ppo.json"
else
  echo "invalid algorithm provided"
  exit 0 
fi

export STUDY_NAME=img_${NORM_BASE}_${ALGO}_${STEPS}_f3
export EVAL_FINAL_EPISODES=10000

if [ $SLURM_ARRAY_TASK_ID -lt 1 ]; then sleep $(( 40 * SLURM_ARRAY_TASK_ID )); fi

python ../train_hyper.py --use-images --algo ${ALGO} --norm-base ${NORM_BASE} --hp-file ${hpfile} --seeds 1 --steps ${STEPS} --save-models --save-video --log-csv
