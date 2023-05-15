# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os
import sys

import easydict
import hydra
import pytorch_lightning as pl
import torch
from dataset import get_collate_fn, get_dataset
from finetune.finetune_wrapper import FinetuningWrapper
from finetune.utils.hydra_utils import initialize_config
from model import get_contextual_relation_network, get_shot_encoder


def init_hydra_config(mode: str):
    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    overrides = sys.argv[1:]
    logging.info(f"####### overrides: {overrides}")
    with hydra.initialize_config_module(config_module="finetune.cfg"):
        cfg = hydra.compose("default", overrides=overrides)

    cfg = initialize_config(cfg, mode=mode)
    return cfg


def apply_random_seed(cfg):
    if "SEED" in cfg and cfg.SEED >= 0:
        pl.seed_everything(cfg.SEED, workers=True)


def load_finetuned_config(cfg):
    ckpt_root = cfg.CKPT_PATH
    load_from = cfg.LOAD_FROM
    with open(os.path.join(ckpt_root, load_from, "config.json"), "r") as fopen:
        finetuned_cfg = json.load(fopen)
        finetuned_cfg = easydict.EasyDict(finetuned_cfg)

    # override configuration of pre-trained model
    cfg.MODEL = finetuned_cfg.MODEL
    # Here finetuned and pre-trained models are the same. Thus their configs
    # cfg.PRETRAINED_LOAD_FROM = finetuned_cfg.PRETRAINED_LOAD_FROM

    cfg.TRAIN.USE_SINGLE_KEYFRAME = False
    cfg.MODEL.contextual_relation_network.params.trn.pooling_method = "center"

    # override neighbor size of an input sequence of shots
    sampling = finetuned_cfg.LOSS.sampling_method.name
    nsize = finetuned_cfg.LOSS.sampling_method.params[sampling]["neighbor_size"]
    cfg.LOSS.sampling_method.params["sbd"]["neighbor_size"] = nsize

    return cfg

def load_pretrained_config(cfg):
    load_from = cfg.PRETRAINED_LOAD_FROM
    ckpt_root = cfg.PRETRAINED_CKPT_PATH

    with open(os.path.join(ckpt_root, load_from, "config.json"), "r") as fopen:
        pretrained_cfg = json.load(fopen)
        pretrained_cfg = easydict.EasyDict(pretrained_cfg)

    # override configuration of pre-trained model
    cfg.MODEL = pretrained_cfg.MODEL
    # set to use contextual relation network
    cfg.MODEL.contextual_relation_network.enabled = True

    # override neighbor size of an input sequence of shots
    sampling = pretrained_cfg.LOSS.sampling_method.name
    cfg.LOSS.sampling_method.params["sbd"][
        "neighbor_size"
    ] = pretrained_cfg.LOSS.sampling_method.params[sampling]["neighbor_size"]

    return cfg


def init_data_loader(cfg, mode, is_train):
    data_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(cfg, mode=mode, is_train=is_train),
        batch_size=cfg.TRAIN.BATCH_SIZE.batch_size_per_proc,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        persistent_workers=is_train,
        drop_last=is_train,
        shuffle=is_train,
        collate_fn=get_collate_fn(cfg),
    )
    if is_train:
        # need for warmup
        cfg.TRAIN.TRAIN_ITERS_PER_EPOCH = (
            len(data_loader.dataset) // cfg.TRAIN.BATCH_SIZE.effective_batch_size
        )
    return cfg, data_loader


def init_model(cfg):
    shot_encoder = get_shot_encoder(cfg)
    crn = get_contextual_relation_network(cfg)
    if "LOAD_FROM" in cfg and len(cfg.LOAD_FROM) > 0:
        print("LOAD SBD MODEL WEIGHTS FROM: ", cfg.LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            shot_encoder=shot_encoder,
            crn=crn,
            checkpoint_path=os.path.join(
                cfg.CKPT_PATH, cfg.LOAD_FROM, "finetune-model-epoch=14.ckpt"),
            strict=False,
        )
    elif "PRETRAINED_LOAD_FROM" in cfg and len(cfg.PRETRAINED_LOAD_FROM) > 0:
        print("LOAD PRETRAINED MODEL WEIGHTS FROM: ", cfg.PRETRAINED_LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            shot_encoder=shot_encoder,
            crn=crn,
            checkpoint_path=os.path.join(
                cfg.PRETRAINED_CKPT_PATH, cfg.PRETRAINED_LOAD_FROM, "model-v1.ckpt"
            ),
            strict=False,
        )
    else:
        model = FinetuningWrapper(cfg, shot_encoder, crn)
    logging.info(f"MODEL: {model}")
    return cfg, model


def init_trainer(cfg):
    # logger
    logger = None
    callbacks = []
    if cfg.MODE == "finetune":
        logs_path = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        os.makedirs(logs_path, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(logs_path, version=0)

        # checkpoint callback
        ckpt_path = os.path.join(cfg.CKPT_PATH, cfg.EXPR_NAME)
        os.makedirs(ckpt_path, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path,
                monitor="sbd_test/ap",
                verbose=True,
                filename="finetune-model-{epoch}",
                save_top_k=2,
                mode="max"
            )
        )
        # early stopping callback
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="sbd_test/ap",
                patience=cfg.TRAIN.PATIENCE,
                verbose=True,
                mode="max",
            )
        )
        # learning rate callback
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
        # GPU stat callback
        callbacks.append(
            pl.callbacks.GPUStatsMonitor(
                memory_utilization=True,
                gpu_utilization=True,
                intra_step_time=True,
                inter_step_time=True,
            )
        )
    trainer = pl.Trainer(**cfg.TRAINER, callbacks=callbacks, logger=logger)
    return cfg, trainer
