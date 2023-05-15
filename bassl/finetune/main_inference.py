#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging, ipdb
import os
import pickle
import torch
import numpy as np

from finetune.utils.hydra_utils import print_cfg
from finetune.utils.main_utils import (
    apply_random_seed,
    init_data_loader,
    init_hydra_config,
    load_finetuned_config,
    init_model,
    init_trainer,
)


def main(save_pred: bool = False)->None:
    # init cfg
    cfg = init_hydra_config(mode="inference")
    apply_random_seed(cfg)
    cfg = load_finetuned_config(cfg)

    # init dataloader
    cfg, data_loader = init_data_loader(cfg, mode="inference", is_train=False)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    # print cfg
    print_cfg(cfg)

    # train
    logging.info(
        f"Start Predicting..."
    )

    pred = torch.hstack(trainer.predict(model, data_loader))
    pred = pred.detach().cpu().numpy()
    # ipdb.set_trace()

    # 'idx': Tensor, 'vid': List[str], 'sid': List[str], 
    # 'nsids': Tensor
    global_dict = {'idx': torch.zeros((0)), 'vid': [],
                   'sid': [], 'nsids': torch.zeros((0, 17))}
    for batch in data_loader:
        for k, v in batch.items():
            if k in ['vid', 'sid']:
                global_dict[k].extend(v)
            elif k in ['idx', 'nsids']:
                global_dict[k] = torch.cat((global_dict[k], v), dim=0)
            elif k == "video":
                continue
            else:
                raise KeyError(f'Unknown key: {k}')
    for k in ['idx', 'nsids']:
        global_dict[k] = global_dict[k].numpy().astype(np.int32)

    if save_pred:
        os.makedirs(cfg.PRED_SAVE_PATH, exist_ok=True)
        print(f"Save prediction to {cfg.PRED_SAVE_PATH}")
        np.save(os.path.join(cfg.PRED_SAVE_PATH, "pred.npy"), pred)
        print(f"Save metadata to {cfg.PRED_SAVE_PATH}")
        with open(os.path.join(cfg.PRED_SAVE_PATH, "pred_metadata.pkl"), "wb") as f:
            pickle.dump(global_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving Done!")

if __name__ == "__main__":
    main(save_pred=True)
