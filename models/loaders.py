from functools import partial

import torch

import ImplicitMotion.models.code_dict as code
import ImplicitMotion.models.motion_model as motion_model
import ImplicitMotion.models.time_function as time

def get_optimizer(optim_opts):
    optim_dict = optim_opts.__dict__.copy()
    del optim_dict["optimizer"]
    return partial(getattr(torch.optim, optim_opts.optimizer), **optim_dict)

def get_filelist_from_ckpt(ckpt):
    key_list = list(ckpt["sequence_code"].keys())

    ckpt_filelist = [f.split('.')[1] for f in key_list if "mean_param" in f]
    if len(ckpt_filelist) == 0:
        ckpt_filelist = [f.split('.')[1] for f in key_list if "param" in f]
    return ckpt_filelist

def get_models(config, filenames, action_labels, device, use_action_label=True):
    #### Sequence code
    if config.sequence_code_opts.variational:
        sequence_code = code.VariationalCodeDict(filenames, "id", config.sequence_code_opts)
    else:
        sequence_code = code.CodeDict(filenames, "id", config.sequence_code_opts)
    
    #### Action code
    if use_action_label:
        action_code_args = (action_labels, "label", config.action_code_opts)
    else:
        action_code_args = (filenames, "id", config.action_code_opts)

    if config.action_code_opts.variational:
        action_code = code.VariationalCodeDict(*action_code_args)
    else:
        action_code = code.CodeDict(*action_code_args)

    #### Time function
    time_function = time.FixedPositionalEmbeddingFunction(
        filenames, "id", config.positional_embedding_opts
    )

    if config.model_type == "mlp":
        decoder = motion_model.ImplicitMotionModel(config)
    elif config.model_type == "transformer":
        decoder = motion_model.TransformerDecoder(config)

    model_dict = {
        "motion_model": decoder.to(device),
        "sequence_code": sequence_code.to(device),
        "action_code": action_code.to(device),
        "time_function": time_function.to(device)
    }

    optim_dict = {}
    optim_dict["motion_model"] = \
        get_optimizer(config.model_optimizer)(model_dict["motion_model"].parameters())

    optim_dict["sequence_code"] = \
        get_optimizer(config.sequence_code_optimizer)(model_dict["sequence_code"].parameters())

    optim_dict["action_code"] = \
        get_optimizer(config.action_code_optimizer)(model_dict["action_code"].parameters())

    return model_dict, optim_dict