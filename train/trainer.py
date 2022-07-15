import logging
import os
import re
import copy


import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import ImplicitMotion.train.train as T
import ImplicitMotion.models.loaders as load
from ImplicitMotion.data.motion_sequences import MotionSequence

from ImplicitMotion.data.visual import Pose3DAnimator, Pose3DStrip
import ImplicitMotion.data.datasets as data
import ImplicitMotion.data.CONSTANTS as K

def get_trainer(config):
    if config.dataset == "Human":
        trainer = HumanActTrainer(config)
    elif config.dataset == "NTU13":
        trainer = NTUVibeTrainer(config)
    elif config.dataset == "UESTC":
        trainer = UESTCTrainer(config)
    return trainer

def barrier():
    # https://github.com/pytorch/pytorch/issues/15051
    t = torch.randn((), device='cuda')
    dist.all_reduce(t)
    torch.cuda.synchronize()

def distributed_alternating_training(config, epoch, dataloader, model_dict, loss_dict, optimizers, device, summary=None):
    torch.cuda.set_device(device)
    code_updates = config.code_update_ratio if config.code_update_ratio > 1 else 1
    model_updates = int(1.0/config.code_update_ratio) if config.code_update_ratio < 1 else 1
    optim_code = ["sequence_code", "action_code", "time_function"]
    for key, optim in optimizers.items():
        optim.zero_grad()
    max_iter = code_updates + model_updates

    for i in range(code_updates):
        log_list = []
        if dist.get_rank() == 0:
            # iterator = dataloader
            logging.info("Epoch %d - Updating codes - %d iteration", epoch, i+1)
            iterator = tqdm(dataloader)
        else:
            iterator = dataloader
        for iter, batch in enumerate(iterator):
            #### forward/backward pass
            log = T.train_iteration(batch, model_dict, loss_dict, config, device, epoch)
            log_list.append(log)

            ### gather gradients
            size = float(dist.get_world_size())

            for key, model in model_dict.items():
                if key in ["action_code"]:
                    for param in model.parameters():
                        if param.grad is None:
                            param.grad = torch.zeros_like(param.data, device=config.device[dist.get_rank()])
                        # dist.barrier()
                        if param.data.numel() > 0:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= size

            # #### optimizer update
            for key, optim in optimizers.items():
                if key in optim_code:
                    if config.gradient_clipping:
                        for pg in optim.param_groups:
                            torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=2.0)
                    optim.step()

        loss_summary = {}

        for key, loss in log_list[0]["loss"].items():
            losses = torch.stack([l["loss"][key].mean() for l in log_list]).mean()
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            loss_summary[key] = losses

        if summary is not None:
            summary.add_scalars("loss", loss_summary, epoch * max_iter + i)

        for key, optim in optimizers.items():
            optim.zero_grad()
        dist.barrier()

    if not all([k in optim_code for k in list(optimizers.keys())]):
        for i in range(model_updates):
            log_list = []
            if dist.get_rank() == 0:
                logging.info("Epoch %d - Updating model - %d iteration", epoch, i+1)
                # iterator = dataloader
                iterator = tqdm(dataloader)
            else:
                iterator = dataloader
            for iter, batch in enumerate(iterator):
                #### forward/backward pass
                log = T.train_iteration(batch, model_dict, loss_dict, config, device, epoch)
                log_list.append(log)

                ### gather gradients
                size = float(dist.get_world_size())
                for key, model in model_dict.items():
                    if key in optim_code: continue
                    for param in model.parameters():
                        if param.data.numel() > 0:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= size

                #### optimizer update
                for key, optim in optimizers.items():
                    if key not in optim_code:
                        if config.gradient_clipping:
                            for pg in optim.param_groups:
                                torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=2.0)
                        optim.step()

                for key, optim in optimizers.items():
                    optim.zero_grad()

            loss_summary = {}

            for key, loss in log_list[0]["loss"].items():
                losses = torch.stack([l["loss"][key].mean() for l in log_list]).mean()
                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                loss_summary[key] = losses

            if summary is not None:
                summary.add_scalars("loss", loss_summary, epoch * max_iter + code_updates + i)

def init_distributed_train(rank, config):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S")

    device = config.device[rank]
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", rank=rank, world_size=len(config.device))

    torch.manual_seed(config.seed) #same seed for each process guarantees same model initialization

    trainer = get_trainer(config)
    loss_dict = trainer.init_loss()

    ### Prepare data
    full_dataloader = trainer.init_dataloader(device, split=config.split)
    if MotionSequence._SMPL is not None:
        MotionSequence._SMPL.to(device) # needs to run after init dataloader or SMPL is not yet initialized
    filelist = full_dataloader.dataset.filelist
    action_labels = full_dataloader.dataset.action_labels
    partitioner = data.DataPartitioner(full_dataloader.dataset, len(config.device))

    dataset = partitioner.use(full_dataloader.dataset, rank)
    part_filelist = [filelist[idx] for idx in dataset.indices]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True, pin_memory=False,
        collate_fn=full_dataloader.dataset.collate_fn
    )

    ### Initialize model
    model_dict, optimizers = trainer.init_model(part_filelist, action_labels, device=device)

    if rank == 0 and config.logging_opts.logging and config.logging_opts.tensorboard:
        summary = SummaryWriter(config.path_results)
    else:
        summary = None

    for epoch in range(config.epochs):
        distributed_alternating_training(config, epoch, dataloader, model_dict, loss_dict, optimizers, device, summary=summary)
        if (epoch + 1) % config.logging_opts.checkpoint_epoch == 0:
            if rank == 0:
                logging.info("Epoch %d - Model stored", epoch)
                checkpoint_state_dict = {
                    "action_code": model_dict["action_code"].state_dict(),
                    "motion_model": model_dict["motion_model"].state_dict(),
                    "optim_action_code": optimizers["action_code"].state_dict(),
                    "optim_motion_model": optimizers["motion_model"].state_dict()
                }
                torch.save(checkpoint_state_dict, trainer.make_checkpoint_name(epoch + 1))
                torch.save(checkpoint_state_dict, trainer.path_checkpoint_base.joinpath("latest.pt"))

            sequence_ckpt = {
                f"sequence_code_{rank}": model_dict["sequence_code"].state_dict(),
                f"optim_sequence_code_{rank}": optimizers["sequence_code"].state_dict()
            }
            torch.save(sequence_ckpt, trainer.path_checkpoint_base.joinpath(f"{config.time}_seq_code_part{rank}_epoch{epoch+1}.pt"))
            torch.save(sequence_ckpt, trainer.path_checkpoint_base.joinpath(f"latest_seq_code_part{rank}_epoch{epoch+1}.pt"))

class Trainer(object):
    def __init__(self, config: dict):
        self.config = config

        if self.config.logging_opts.logging and self.config.logging_opts.tensorboard and not isinstance(self.config.device, list):
            self.summary = SummaryWriter(self.config.path_results)

        if self.config.logging_opts.logging and self.config.logging_opts.checkpoint:
            self.path_checkpoint_base = self.config.path_results.joinpath("checkpoints")
            if not self.path_checkpoint_base.exists():
                self.path_checkpoint_base.mkdir()

    def train(self):
        dataloader = self.init_dataloader(self.config.device, split=self.config.split)
        loss_dict = self.init_loss()

        filelist = dataloader.dataset.filelist
        action_labels = dataloader.dataset.action_labels

        if self.config.load_checkpoint:
            model_dict, optimizers = self.init_model(
                filelist, action_labels, device=self.config.device, path_ckpt=self.config.path_checkpoint, epoch=self.config.checkpoint_epoch)
            if self.config.continue_training:
                start_epoch = self.config.checkpoint_epoch
            else:
                start_epoch = 0
        else:
            model_dict, optimizers = self.init_model(filelist, action_labels, device=self.config.device)
            start_epoch = 0

        for epoch in range(start_epoch, self.config.epochs):
            logging.info("Training epoch %d - for %s", epoch, self.config.time)
            for model in model_dict.values():
                model.train()

            self.train_epoch(epoch, dataloader, model_dict, loss_dict, optimizers, self.config.device)

            if self.config.logging_opts.logging and self.config.logging_opts.checkpoint:
                if (epoch + 1) % self.config.logging_opts.checkpoint_epoch == 0:
                    self.store_checkpoint(epoch + 1, model_dict, optimizers)

            if self.config.logging_opts.logging and self.config.logging_opts.testing:
                self.test_epoch(epoch, dataloader, model_dict, loss_dict, self.config.device)

    def make_checkpoint_name(self, epoch):
        return self.path_checkpoint_base.joinpath(f"{self.config.time}_epoch{epoch}.pt")

    def train_epoch(self, *args):
        if self.config.optim_alternating:
            self.alternating_training(*args)
        else:
            self.normal_training(*args)

    def test_epoch(self, epoch: int, dataloader, model_dict, loss_dict, device):
        logging.info("Testing epoch %d", epoch)
        modified_config = copy.deepcopy(self.config)
        modified_config["time_SGD"] = False
        with torch.no_grad():
            losses = []
            for iter, motion_sequences in enumerate(dataloader["code"]):
                batch_data = T.data_preprocessing(motion_sequences, modified_config, device)
                prediction = T.predict_motion(model_dict, batch_data, epoch=epoch)
                losses.append(loss_dict["test"]["recon"](**batch_data, **prediction))

            print(torch.stack(losses).mean().item())
            self.summary.add_scalar("test", torch.stack(losses).mean().item(), epoch)

    def aggregate_logs(self, log_list):
        epoch_log = {}
        for type, log_dict in log_list[0].items():
            epoch_log[type] = {k: torch.cat([f[type][k] for f in log_list]).mean() for k in log_dict.keys()}
        return epoch_log

    def summary_add_log(self, log, niter):
        for k, v in log.items():
            if isinstance(v, dict):
                self.summary.add_scalars(k, v, niter)
            else:
                self.summary.add_scalar(k, v, niter)

    def distributed_train(self):
        logging.info("Starting distributed training")
        mp.set_start_method("forkserver")
        mp.set_sharing_strategy('file_system')

        processes = []
        for rank, device in enumerate(self.config.device):
            p = mp.Process(target=init_distributed_train, args=(rank,self.config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def alternating_training(self, epoch, dataloader, model_dict, loss_dict, optimizers, device):
        code_updates = self.config.code_update_ratio if self.config.code_update_ratio > 1 else 1
        model_updates = int(1.0/self.config.code_update_ratio) if self.config.code_update_ratio < 1 else 1
        optim_code = ["sequence_code", "action_code", "time_function"]

        max_iter = code_updates + model_updates

        for i in range(code_updates):
            logging.info("Epoch %d - Updating codes - %d iteration", epoch, i+1)
            log_list = []
            for iter, batch in enumerate(tqdm(dataloader)):
                #### forward/backward pass
                log = T.train_iteration(batch, model_dict, loss_dict, self.config, device, epoch)
                log_list.append(log)

                #### optimizer update
                for key, optim in optimizers.items():
                    if key in optim_code:
                        if self.config.gradient_clipping:
                            for pg in optim.param_groups:
                                torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=2.0)
                        optim.step()

                for key, optim in optimizers.items():
                    optim.zero_grad()

            if self.config.logging_opts.logging and self.config.logging_opts.tensorboard:
                epoch_log = self.aggregate_logs(log_list)
                self.summary_add_log(epoch_log, epoch * max_iter + i)

        if not all([k in optim_code for k in list(optimizers.keys())]):
            for i in range(model_updates):
                log_list = []
                logging.info("Epoch %d - Updating model - %d iteration", epoch, i+1)
                for iter, batch in enumerate(tqdm(dataloader)):
                    #### forward/backward pass
                    log = T.train_iteration(batch, model_dict, loss_dict, self.config, device, epoch)
                    log_list.append(log)

                    #### optimizer update
                    if not self.config.aggregate_per_epoch:
                        for key, optim in optimizers.items():
                            if key not in optim_code:
                                if self.config.gradient_clipping:
                                    for pg in optim.param_groups:
                                        torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=2.0)
                                optim.step()

                        for key, optim in optimizers.items():
                            optim.zero_grad()

                if self.config.aggregate_per_epoch:
                    for key, optim in optimizers.items():
                        if key not in optim_code:
                            if self.config.gradient_clipping:
                                for pg in optim.param_groups:
                                    torch.nn.utils.clip_grad_norm_(pg["params"], max_norm=2.0)
                            optim.step()

                    for key, optim in optimizers.items():
                        optim.zero_grad()

                if self.config.logging_opts.logging and self.config.logging_opts.tensorboard:
                    epoch_log = self.aggregate_logs(log_list)
                    self.summary_add_log(epoch_log, epoch * max_iter + code_updates + i)

    def normal_training(self, epoch, dataloader, model_dict, loss_dict, optimizers, device):
        for iter, batch in enumerate(tqdm(dataloader)):
            #### forward/backward pass
            log = T.train_iteration(batch, model_dict, loss_dict, self.config, device, epoch)

            if self.config.logging_opts.logging and self.config.logging_opts.tensorboard:
                niter = epoch * len(dataloader) + iter
                for k, v in log.items():
                    if isinstance(v, dict):
                        self.summary.add_scalars(k, v, niter)
                    else:
                        self.summary.add_scalar(k, v, niter)

            #### optimizer update
            for key, optim in optimizers.items():
                optim.step()

            for key, optim in optimizers.items():
                optim.zero_grad()

    def store_checkpoint(self, epoch, model_dict, optimizers):
        checkpoint_state_dict = {}
        for key, model in model_dict.items():
            checkpoint_state_dict[key] = model.state_dict()

        for key, optim in optimizers.items():
            checkpoint_state_dict[f"optim_{key}"] = optim.state_dict()
        
        torch.save(checkpoint_state_dict, self.make_checkpoint_name(epoch))
        torch.save(checkpoint_state_dict, self.path_checkpoint_base.joinpath(f"latest.pt"))
        logging.info("Epoch %d - Model stored", epoch)

    def load_checkpoint(self, path_ckpt, device, model_dict, optimizers, epoch = -1):
        if isinstance(device, int):
            device = f"cuda:{device}"

        if path_ckpt.is_file():
            ckpt = torch.load(path_ckpt, map_location=device)
        elif path_ckpt.exists():
            if epoch == -1:
                raise ValueError("If path_ckpt is a directory, you need to specify a valid epoch. epoch == -1")
            check_epoch = lambda x: re.search("(?<=epoch)\d+", str(x))[0] == str(epoch)
            ckpt_files = list(filter(check_epoch,path_ckpt.joinpath("checkpoints").iterdir()))
            
            if len(ckpt_files) == 0:
                raise ValueError("No checkpoint for epoch %d", epoch)
            ckpt = {}
            for file in ckpt_files:
                ckpt.update(torch.load(file, map_location=device))
        else:
            raise ValueError("%s is not a valid path", path_ckpt)
        
        for key, model in model_dict.items():
            try:
                model.load_state_dict(ckpt[key])
            except KeyError:
                for k in ckpt.keys():
                    if re.match(f"{key}_\d+", k):
                        model.load_state_dict(ckpt[k], strict=False)

        for key, optim in optimizers.items():
            if f"optim_{key}" in ckpt and isinstance(ckpt[f"optim_{key}"], dict):
                optim.load_state_dict(ckpt[f"optim_{key}"])
            else:
                logging.warn("Optimizer for %s not found in checkpoint. Continue anyway", key)

    def rotation_loss(self, *, prediction, groundtruth, **kwargs):
        rot_loss = torch.nn.functional.mse_loss(
            torch.cat(prediction[self.config.model_opts.motion_representation]),
            torch.cat(groundtruth[self.config.model_opts.motion_representation]))
        return rot_loss.unsqueeze(0)

    # def vertices_loss(self, *, prediction, groundtruth, **kwargs):
    #     # somehow negatively affects training - do not use
    #     vert_loss = torch.nn.functional.mse_loss(
    #         checkpoint.checkpoint(prediction.batch_conversion,self.config.model_opts.motion_representation, "vertices_noroot"),
    #         groundtruth.batch_conversion(self.config.model_opts.motion_representation, "vertices_noroot"))
    #     return vert_loss.unsqueeze(0)

    def vertices_loss(self, *, prediction, groundtruth, **kwargs):
        vert_loss = torch.nn.functional.mse_loss(
            prediction.batch_conversion(self.config.model_opts.motion_representation, "vertices_noroot"),
            groundtruth.batch_conversion(self.config.model_opts.motion_representation, "vertices_noroot"))
        return vert_loss.unsqueeze(0)

    def root_loss(self, *, prediction, groundtruth, **kwargs):
        root_loss = torch.nn.functional.mse_loss(
            torch.cat(prediction["root"]),
            torch.cat(groundtruth["root"]))
        return root_loss.unsqueeze(0)

    def joint_loss(self, *, prediction, groundtruth, **kwargs):
        tgt_joint_locs = torch.cat(groundtruth["joints"])
        tgt_root_locs = torch.cat(groundtruth["root"])
        tgt_joints = torch.cat([tgt_root_locs, tgt_joint_locs], 1)
        pd_joint_locs = torch.cat(prediction.forward_kinematics(joints=tgt_joints))
        return torch.nn.functional.mse_loss(tgt_joints, pd_joint_locs).unsqueeze(0)

    def init_loss(self):
        loss_dict = {}

        if self.config.recon_loss_type == "joint":
            loss_dict["recon_joint"] = self.joint_loss
        elif self.config.recon_loss_type == "vertices":
            loss_dict["recon_vert"] = self.vertices_loss
        elif self.config.recon_loss_type == "rotation":
            loss_dict["recon_rot"] = self.rotation_loss
        elif self.config.recon_loss_type == "combined":
            loss_dict["recon_rot"] = self.rotation_loss
            loss_dict["recon_vert"] = self.vertices_loss
            loss_dict["recon_root"] = self.root_loss
        elif self.config.recon_loss_type == "combined_joints":
            loss_dict["recon_rot"] = self.rotation_loss
            loss_dict["recon_joint"] = self.joint_loss
        else:
            raise NotImplementedError("Loss function %s not implemented", self.config.recon_loss_type)

        if self.config.sequence_code_opts.variational:
            loss_dict["sequence_kld"] = lambda **kwargs: kwargs["sequence_code"]["kld"]

        if self.config.action_code_opts.variational:
            loss_dict["action_kld"] = lambda **kwargs: kwargs["action_code"]["kld"]
        return loss_dict

    def init_model(self, filenames, action_labels, use_action_label=True, device="cuda:0", path_ckpt = None, epoch = -1):
        device = self.config.device if device is None else device
        model_dict, optim_dict = load.get_models(
            self.config, filenames, action_labels, device, use_action_label=use_action_label)

        if path_ckpt is not None:
            self.load_checkpoint(path_ckpt, device, model_dict, optim_dict, epoch=epoch)
        return model_dict, optim_dict

class HumanActTrainer(Trainer):
    def __init__(self, config):
        self.classifier_filename = 'action_recognition_model_humanact12.tar'
        self.K = {
            "A2I": K.HUMAN_A2I,
            "I2A": K.HUMAN_I2A,
            "default_bones": K.HUMAN_default_bones,
            "bone_hierarchy": K.HUMAN_bone_hierarchy
        }
        super().__init__(config)

    def get_animator(self, *args, animation_type="anim", **kwargs):
        if animation_type == "anim":
            return Pose3DAnimator(K.HUMAN_bone_hierarchy, *args, **kwargs)
        elif animation_type == "strip":
            return Pose3DStrip(K.HUMAN_bone_hierarchy, *args, **kwargs)
        else:
            raise ValueError("Animation type %s not known",animation_type)

    def init_dataloader(self, device, ckpt_filelist=None, **kwargs):
        dataset = data.HumanAct12(
            self.config.path_dataset, 
            device=device, 
            sample_ids=ckpt_filelist,
            num_workers=self.config.num_workers,
            path_smpl=self.config.path_smpl,
            use_SMPL=self.config.recon_loss_type in ["vertices", "rotation", "combined", "combined_joints", "joint"],
            **kwargs)

        if self.config.recon_loss_type in ["vertices", "rotation", "combined", "combined_joints"]:
            self.SMPL_MODEL = dataset.SMPL_MODEL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True, pin_memory=False,
            collate_fn=dataset.collate_fn
        )

        return dataloader


class NTUVibeTrainer(Trainer):
    def __init__(self, config):
        self.classifier_filename = 'action_recognition_model_vibe_v2.tar'

        sorted_labels = [5, 6, 7, 8, 21, 22, 23, 37, 79, 92, 98, 99, 101]
        map_new_labels = {j:i for i,j in zip(sorted_labels, list(range(len(sorted_labels))))}

        NTU_I2A = {i: K.NTU_VIBE_I2A[map_new_labels[i]] for i in range(len(sorted_labels))}
        NTU_A2I = {v:k for k, v in NTU_I2A.items()}
        self.K = {
            "A2I": NTU_A2I,
            "I2A": NTU_I2A,
            "default_bones": K.NTU_VIBE_default_bones,
            "bone_hierarchy": K.NTU_VIBE_bone_hierarchy
        }
        super().__init__(config)

    def get_animator(self, *args, **kwargs):
        return Pose3DAnimator(K.NTU_VIBE_bone_hierarchy, *args, **kwargs)

    def init_dataloader(self, device, split="full", **kwargs):
        dataset = data.NTUVIBE13(
            self.config.path_dataset,
            device=device,
            split=split)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True, pin_memory=False,
            collate_fn=dataset.collate_fn
        )

        return dataloader


class UESTCTrainer(Trainer):
    def __init__(self, config):
        self.classifier_filename = 'uestc_rot6d_stgcn.tar'
        self.splits = ["train", "test"]
        self.K = {
            "A2I": K.UESTC_A2I,
            "I2A": K.UESTC_I2A,
        }
        super().__init__(config)

    def init_dataloader(self, device, ckpt_filelist=None, exclude_actions=None, **kwargs):
        # if exclude_actions is None:
        #     exclude_actions = self.config.exclude_actions

        dataset = data.UESTC(
            self.config.path_dataset, 
            device=device, 
            sample_ids=ckpt_filelist,
            num_workers=self.config.num_workers,
            chunk_limit=self.config.chunk_limit,
            approx_chunk_size=self.config.approx_chunk_size,
            path_smpl=self.config.path_smpl,
            use_SMPL=True,
            # exclude_actions=self.config.exclude_actions,
            **kwargs)

        self.SMPL_MODEL = dataset.SMPL_MODEL

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True, pin_memory=False,
            collate_fn=dataset.collate_fn
        )

        return dataloader

    def joint_loss(self, *, prediction, groundtruth, **kwargs):
        joint_loss = torch.nn.functional.mse_loss(
            prediction.batch_conversion(self.config.model_opts.motion_representation, "joints_noroot"),
            groundtruth.batch_conversion("axis_angle", "joints_noroot"))
        return joint_loss.unsqueeze(0)

    def get_animator(self, *args, animation_type="anim", **kwargs):
        if animation_type == "anim":
            return Pose3DAnimator(K.UESTC_bone_hierarchy, *args, **kwargs)
        elif animation_type == "strip":
            return Pose3DStrip(K.UESTC_bone_hierarchy, *args, **kwargs)
        else:
            raise ValueError("Animation type %s not known",animation_type)
