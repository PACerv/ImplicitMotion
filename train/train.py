import torch

import ImplicitMotion.data.motion_sequences as motion

def data_preprocessing(motion_sequences, config, device):
    #### Time batch construction
    if config.batch_subsampling == "random":
        motion_sequences = [motion_sequence.get_subsample(config.batch_subsample_size) for motion_sequence in motion_sequences]
    elif config.batch_subsampling == "fixed_length":
        motion_sequences = [motion_sequence.get_subsequence(config.batch_subsample_size) for motion_sequence in motion_sequences]
    elif config.batch_subsampling == "full":
        pass
    else:
        raise NotImplementedError("Sampling strategy %s unknown", config.batch_subsampling)

    motion_batch = motion.BatchMotionSequence(motion_sequences)

    return motion_batch

def train_iteration(motion_sequences, model_dict, loss_dict, config, device, epoch):
    log = {}

    #### Data loading
    motion_batch = data_preprocessing(motion_sequences, config, device)

    prediction = predict_motion(config, model_dict, motion_batch, epoch=epoch)

    loss_sum = 0
    log["loss"] = {}
    for key, loss in loss_dict.items():
        if key == "test": continue
        log["loss"][key] = loss(groundtruth=motion_batch, **prediction)
        loss_sum = loss_sum + log["loss"][key].mean()

    metrics = {}

    if config.action_code_opts.variational:
        action_code_mean, action_code_var = model_dict["action_code"].get_stats(motion_batch)
        metrics["action_code_var"] = action_code_var.norm(2,-1).mean().unsqueeze(0)
    else:
        action_code_mean = model_dict["action_code"].get_stats(motion_batch)
        action_code_var = None
    metrics["action_code_mean"] = action_code_mean.norm(2,-1).mean().unsqueeze(0)

    if config.sequence_code_opts.variational:
        sequence_code_mean, sequence_code_var = model_dict["sequence_code"].get_stats(motion_batch)
        metrics["sequence_code_var"] = sequence_code_var.norm(2,-1).mean().unsqueeze(0)
    else:
        sequence_code_mean = model_dict["sequence_code"].get_stats(motion_batch)
    metrics["sequence_code_mean"] = sequence_code_mean.norm(2,-1).mean().unsqueeze(0)

    if len(metrics) != 0:
        log["metrics"] = metrics

    loss_sum.backward()

    return log

def predict_motion(config, model_dict, motion_batch, epoch=0):
    #### Assemble code
    sequence_code = model_dict["sequence_code"](motion_batch,epoch=epoch)
    action_code = model_dict["action_code"](motion_batch, epoch=epoch)

    if config.action_code_additive:
        code = sequence_code["code"] + action_code["code"]
    else:
        code = torch.cat([sequence_code["code"], action_code["code"]], 1)

    #### Predict time function
    time_sequences = model_dict["time_function"](motion_batch, epoch=epoch)

    prediction = model_dict["motion_model"](time_sequences["time_sequences"], code, epoch=epoch)
    output = {
        "prediction": prediction,
        "sequence_code": sequence_code,
        "action_code": action_code,
        "time_sequences": time_sequences
    }

    return output
