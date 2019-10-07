#!/bin/bash
# You need to modify this path
DATASET_DIR="/home/r506/DCase/data"

# You need to modify this path as your workspace
WORKSPACE="/home/r506/hhy"


DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"
BACKEND="pytorch"	# "pytorch" | "keras"
HOLDOUT_FOLD=1
GPU_ID=0


############ Extract features ############
#python utils/feature.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
############ Development subtask A ############
# Train model for subtask A
#CUDA_VISIBLE_DEVICES=$GPU_ID python model/main.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD
#CUDA_VISIBLE_DEVICES=$GPU_ID python model/main.py train  --validate --holdout_fold=$HOLDOUT_FOLD

## Evaluate subtask A
CUDA_VISIBLE_DEVICES=$GPU_ID python model/main.py inference_validation_data  --holdout_fold=$HOLDOUT_FOLD --iteration=5000

############ Full train subtask A ############
# #Train on full development data
#CUDA_VISIBLE_DEVICES=$GPU_ID python model/main.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda


#python test.py  --workspace=$WORKSPACE #问题就是os.path.join()无法正常将shell里的参数传入main中，形成合法的路径
