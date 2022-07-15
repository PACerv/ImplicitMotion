import pathlib
import logging
import yaml
import datetime
from dataclasses import dataclass

import torch
import shutil

from ImplicitMotion.data.datasets import get_dataset_opts

def code_snapshot(config, path):
    path_snapshot = config["path_results"].joinpath("snapshot")
    path_snapshot.mkdir()

    if path in path_snapshot.resolve().parents:
        logging.warning("Snapshot path contains itself. No snapshot made %s", path)
    else:
        logging.info("Creating snapshot of %s", path)
        shutil.make_archive(path_snapshot.joinpath(pathlib.Path(path).parts[-1]) ,"zip", path)

        path_config = config["path_config"]
        shutil.copy(path_config, path_snapshot.joinpath(path_config.name))
        logging.info("Creating snapshot of config file %s", str(path_config.absolute()))

def get_config(path, mkdir=False):
    logging.info("Loading config from %s", path)
    with open(path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config["path_config"] = path

    config["logging_opts"] = LoggingOpts(**config["logging_opts"])

    # Select model config
    if config["model_type"] == "transformer":
        config["model_opts"] = TransformerModelOpts(**config["model_opts"], **config["model_transformer"])
    elif config["model_type"] == "mlp":
        config["model_opts"] = MLPModelOpts(**config["model_opts"], **config["model_mlp"])
    else:
        raise ValueError(f"{config['model_type']} unknown model type")

    del config["model_transformer"]
    del config["model_mlp"]

    config["positional_embedding_opts"] = PositionalEmbeddingOpts(**config["positional_embedding_opts"])

    config["dataset_opts"] = DatasetOpts(**get_dataset_opts(config["dataset"]))

    for key, val in config.items():
        if "path" in key:
            config[key] = pathlib.Path(val)

        if "optimizer" in key:
            config[key] = Optimizer(**val)

        if "code_opts" in key:
            config[key] = CodeDictOpts(**val)

    config["time"] = datetime.datetime.now().strftime("%y%m%d_%H%M")

    #### Prepare result folder
    if config["load_checkpoint"]:
        # logging.info("Loading config at %s", config["path_checkpoint"].joinpath("snapshot").joinpath("config.yaml"))
        old_config = get_config(config["path_checkpoint"].joinpath("snapshot").joinpath("config.yaml")).__dict__
        old_config["time"] = "_".join(config["path_checkpoint"].stem.split("_")[0:2])

        if config["continue_training"]:
            config["path_results"] = config["path_results_base"].joinpath(old_config["time"])
        else:
            config["path_results"] = config["path_results_base"].joinpath(config["time"])

        del old_config["path_results"]
        del old_config["device"]
        del old_config["load_checkpoint"]
        del old_config["checkpoint_epoch"]
        del old_config["logging_opts"]
        del old_config["time"]
        del old_config["num_workers"]
        del old_config["path_checkpoint"]
        del old_config["continue_training"]
        del old_config["batch_size"]

        for k,v in old_config.items():
            config[k] = v

    if config["logging_opts"].logging and not config["continue_training"]:
        path_results = config["path_results_base"].joinpath(config["time"])
        cnt = 0
        while path_results.exists():
            cnt += 1
            path_results = config["path_results_base"].joinpath(config["time"] + f"_{cnt}")

        if cnt == 0:
            path_results = config["path_results_base"].joinpath(config["time"])
        if mkdir:
            path_results.mkdir()
            logging.info("Create results folder: %s", str(path_results))
        config["path_results"] = path_results
    elif config["logging_opts"].logging and config["continue_training"]:
        pass
    else:
        config["path_results"] = None

    #### Make snapshot
    if config["logging_opts"].logging and config["logging_opts"].snapshot and not (config["load_checkpoint"] and config["continue_training"]) and mkdir:
        code_snapshot(config, pathlib.Path(__file__).parent.parent)

    # for key, val in config.items():
    #     if key in opts:
    #         config[key] = getattr(opts, key)

    return Config(**config)

@dataclass
class ModelOpts:
    motion_representation: str # axis_angle, rot_6D

@dataclass
class TransformerModelOpts(ModelOpts):
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1

@dataclass
class MLPModelOpts(ModelOpts):
    layers: "list[int]"
    root_model_layers: "list[int]"
    root_model: bool = False
    bias: bool = True
    batch_norm: bool = False

@dataclass
class PositionalEmbeddingOpts:
    num_freq: float
    num_dims: int
    additive: bool

@dataclass
class Optimizer:
    optimizer: str = "Adam"
    lr: float = 0.0001
    weight_decay: float = 0.0

@dataclass
class CodeDictOpts:
    num_dims: int
    logvar_scale: float = -10.0
    variational: bool = True
    variational_weight: float = 0.00001

@dataclass
class LoggingOpts:
    logging: bool = True
    snapshot: bool = True
    testing: bool = True
    tensorboard: bool = True
    checkpoint: bool = True
    checkpoint_epoch: int = 100

@dataclass
class DatasetOpts:
    num_joints: int
    num_labels: int

@dataclass
class Config:
    # Dataset settings
    dataset: str
    dataset_opts: DatasetOpts
    path_dataset: pathlib.Path
    path_smpl: pathlib.Path
    split: str
    chunk_limit: int #larger sequences will be chunked
    approx_chunk_size: int #how large will the chunks be

    # System settings
    logging_opts: LoggingOpts
    path_results_base: pathlib.Path
    path_results: pathlib.Path
    path_config: pathlib.Path
    device: str #multi GPU training not implemented yet
    time: str
    num_workers: int

    # Training settings
    load_checkpoint: bool
    path_checkpoint: pathlib.Path
    continue_training: bool
    checkpoint_epoch: int
    seed: int
    epochs: int
    batch_size: int
    batch_subsampling: str # full, fixed_length, random
    batch_subsample_size: int
    aggregate_per_epoch: bool
    gradient_clipping: bool
    optim_alternating: bool
    code_update_ratio: int

    # Loss settings
    recon_loss_type: str #(joint/vertices/rotation/combined)

    # Model settings
    model_type: str # transformer, mlp
    model_opts: ModelOpts
    model_optimizer: Optimizer

    # Embedding function
    positional_embedding_type: str # fixed
    positional_embedding_opts: PositionalEmbeddingOpts

    # Sequence code
    sequence_code_opts: CodeDictOpts
    sequence_code_optimizer: Optimizer

    # Action code
    action_code_additive: bool
    action_code_opts: CodeDictOpts
    action_code_optimizer: Optimizer




