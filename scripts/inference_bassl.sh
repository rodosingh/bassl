#!/usr/bin/env bash
# script for fine-tuning BaSSL

LOAD_FROM=bassl
# WORK_DIR=$(pwd)
WORK_DIR=/home2/rodosingh/PROJECT/bassl/bassl
python=/home2/rodosingh/virtualenvs/BAS/bin/python
save_dir=/ssd_scratch/cvit/rodosingh/data/24/bassl/

# # extract shot representation - using script in pretrain folder
# PYTHONPATH=${WORK_DIR} ${python} ${WORK_DIR}/pretrain/extract_shot_repr.py \
# 		config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
# 	    +config.LOAD_FROM=${LOAD_FROM}
# sleep 10s

# finetune the model
EXPR_NAME=${LOAD_FROM}
PYTHONPATH=${WORK_DIR} ${python} ${WORK_DIR}/finetune/main_inference.py \
	config.LOAD_FROM=${EXPR_NAME} \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.TRAIN.NUM_WORKERS=8 \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.EXPR_NAME=${EXPR_NAME} \
	+config.PRETRAINED_LOAD_FROM="" \
	+config.PRED_SAVE_PATH=${save_dir}

# For new args, just give a `+` prefix to the arg name 
# and then give the value as shown above.
