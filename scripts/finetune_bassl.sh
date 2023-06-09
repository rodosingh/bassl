#!/bin/bash
#SBATCH -A rodosingh
#SBATCH -c 38
#SBATCH --gres=gpu:4
#SBATCH -w gnode079
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home2/rodosingh/PROJECT/bassl/logs/finetune.log
#SBATCH --mail-user aditya.si@research.iiit.ac.in
#SBATCH --mail-type ALL

cd /home2/rodosingh/PROJECT/bassl/scripts

# script for fine-tuning BaSSL
LOAD_FROM=bassl
WORK_DIR=/home2/rodosingh/PROJECT/bassl/bassl
python=/home2/rodosingh/virtualenvs/BAS/bin/python

# extract shot representation
# PYTHONPATH=${WORK_DIR} ${python} ${WORK_DIR}/pretrain/extract_shot_repr.py \
# 		config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
# 	    +config.LOAD_FROM=${LOAD_FROM}
# sleep 10s

# finetune the model
EXPR_NAME=${LOAD_FROM}
PYTHONPATH=${WORK_DIR} ${python} ${WORK_DIR}/finetune/main.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=8 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.EXPR_NAME=${EXPR_NAME} \
	config.WANDB_LOGGING=True \
	config.WANDB_RUN_NAME=finetune_3 \
	+config.PRETRAINED_LOAD_FROM=${LOAD_FROM}
