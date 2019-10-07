#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/r506/DCase/data"

# You need to modify this path as your workspace
WORKSPACE="/home/r506/hhy/workspace"

DEV_SUBTASK_A_DIR="TAU-urban-acoustic-scenes-2019-development"
LB_SUBTASK_A_DIR="TAU-urban-acoustic-scenes-2019-leaderboard"

BACKEND="keras"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE

############ Development subtask A ############
# Train model for subtask A
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train  --alpha=0.2  --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train  --alpha=0.2  --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Evaluate subtask A
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=10100 --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_data_to_truncation  --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=10100 --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_leaderboard_truncation_data  --dev_subdir=$DEV_SUBTASK_A_DIR --leaderboard_subdir=$LB_SUBTASK_A_DIR  --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=7500 --cuda
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_development_truncation_data  --dev_subdir=$DEV_SUBTASK_A_DIR --leaderboard_subdir=$LB_SUBTASK_A_DIR  --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=7500 --cuda

############ Full train subtask A ############
# Train on full development data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --alpha=0.2  --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda

# Inference leaderboard data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_leaderboard_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --leaderboard_subdir=$LB_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=17000 --cuda