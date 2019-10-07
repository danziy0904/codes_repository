#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/r506/DCase/data"


# You need to modify this path as your workspace
WORKSPACE="/home/r506/hhy/workspace"


DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"
#DEV_SUBTASK_A_DIR="TAU-urban-acoustic-scenes-2019-development"
#DEV_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-development"
LB_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-leaderboard"
#LB_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-leaderboard"
#EVAL_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-evaluation"
#EVAL_SUBTASK_B_DIR="TUT-urban-acoustic-scenes-2018-mobile-evaluation"

BACKEND="keras"	# "pytorch" | "keras"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
python utils/feature.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE
