import logging
import torch
import argparse

import numpy as np

from ImplicitMotion.utils.config import get_config
from ImplicitMotion.train.trainer import get_trainer

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_config", default="./ImplicitMotion/config_test.yaml", help="path to config file")
    opts = parser.parse_args()

    config = get_config(opts.path_config, mkdir=True)

    # torch.autograd.set_detect_anomaly(True)
    # torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = get_trainer(config)
    if isinstance(config.device, list):
        trainer.distributed_train()
    else:
        trainer.train()