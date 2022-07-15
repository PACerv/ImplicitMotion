import logging
import pathlib
import itertools
import sys
import re
import argparse
import math

import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import ImplicitMotion.test.a2m_scripts as a2m
import ImplicitMotion.test.actor_scripts as actor

from ImplicitMotion.data.motion_sequences import MotionSequence, BatchMotionSequence
from ImplicitMotion.train import train
from ImplicitMotion.test import test_scripts as test
from ImplicitMotion.test import test_distribution_fitting as dist
from ImplicitMotion.utils.config import get_config
from ImplicitMotion.train.trainer import get_trainer


class EvaluationMetrics(object):
    def __init__(self, test_config, train_dataloader=None, model_dict=None, init_logging=True, split=None):
        self.test_config = test_config
        ### Setup paths
        self.path_results = pathlib.Path(self.test_config["path_results"])
        self.path_eval = self.path_results.joinpath("eval")
        if not self.path_eval.exists():
            self.path_eval.mkdir()

        ### Modify config
        self.config = get_config(str(self.path_results.joinpath("snapshot").joinpath("config.yaml")))
        self.config.device = test_config["device"]
        self.config.logging_opts.logging = False

        if init_logging:
            ### Setup logging
            self.path_log = self.path_eval.joinpath(f"{self.config.time}.txt")
            i=0
            while self.path_log.exists():
                self.path_log = self.path_eval.joinpath(f"{self.config.time}_{i}.txt")
                i+=1

            logging.root.handlers = []
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    logging.FileHandler(str(self.path_log)),
                    logging.StreamHandler(sys.stdout)
            ])

            for key, val in self.test_config.items():
                if isinstance(val, dict):
                    logging.info(key + "\n" + "\n    ".join([f"{k}: {v}" for k, v in val.items()]))
                else:
                    logging.info(f"{key}: {val}")

        self.trainer = get_trainer(self.config)

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.config.seed)
        np.random.seed(seed=self.config.seed)
        # self.config.chunk_limit = -1

        ### Load datasets
        # On UESTC long samples may be chunked into smaller chunks
        # For evaluation the un-chunked dataset should be used
        # We need the chunked dataset to recover the trained model
        if train_dataloader is None:
            if self.config.chunk_limit != -1:
                if split is None:
                    chunked_train_dataset = self.trainer.init_dataloader(test_config["device"], split=self.config.split).dataset
                else:
                    chunked_train_dataset = self.trainer.init_dataloader(test_config["device"], split=split).dataset
                self.chunked_filelist = chunked_train_dataset.filelist
                self.chunked_action_labels = chunked_train_dataset.action_labels
                self.chunked_action_dict = chunked_train_dataset.action_dict
            
            self.config.chunk_limit = -1
            if split is None:
                train_dataloader = self.trainer.init_dataloader(test_config["device"], split=self.config.split)
            else:
                train_dataloader = self.trainer.init_dataloader(test_config["device"], split=split)

        self.filelist = train_dataloader.dataset.filelist
        self.action_labels = train_dataloader.dataset.action_labels
        self.action_dict = train_dataloader.dataset.action_dict

        self.train_dataset = train_dataloader.dataset
        if hasattr(train_dataloader.dataset, "sample_ids"):
            self.sample_ids = train_dataloader.dataset.sample_ids
        else:
            self.sample_ids = None

        self.classifier = self.load_classifier()
        self.model_dict = model_dict

    def load_classifier(self):
        ### Load classifier
        self.path_classifier = pathlib.Path(self.test_config["path_classifier"]).joinpath(self.trainer.classifier_filename)
        if self.config.dataset in ["Human", "NTU13"]:
            classifier = a2m.MotionDiscriminator(self.config.dataset_opts.num_joints * 3, 128, 2, output_size=self.config.dataset_opts.num_labels)
            classifier_state_dict = torch.load(self.path_classifier)
            classifier.load_state_dict(classifier_state_dict["model"])
        elif self.config.dataset in ["UESTC"]:
            classifier = actor.STGCN(
                6, 40, 
                graph_args={"layout": "smpl", "strategy": "spatial"},
                edge_importance_weighting=True,
                device=self.test_config["device"])
            classifier_state_dict = torch.load(self.path_classifier)
            classifier.load_state_dict(classifier_state_dict)
            
        else:
            raise NotImplementedError()
        classifier.eval()
        return classifier.to(self.config.device)

    def evaluate_reconstruction(self, epoch, model_dict=None, test_dataset=None):
        if test_dataset is None:
            groundtruth_dataset = self.train_dataset
        else:
            groundtruth_dataset = test_dataset

        loss_dict = self.trainer.init_loss()
        del_k = []
        for k in loss_dict.keys():
            if not "recon" in k:
                del_k.append(k)

        for k in del_k:
            del loss_dict[k]

        losses = {k: [] for k in loss_dict.keys()}
        if model_dict is None:
            model_dict = self.get_model_dict(epoch)

        train_dataloader = torch.utils.data.DataLoader(
            groundtruth_dataset,
            batch_size=2,
            # sampler=torch.utils.data.RandomSampler(groundtruth_dataset, replacement=True, num_samples=test_config["num_samples"]),
            shuffle=True,
            collate_fn=groundtruth_dataset.collate_fn
        )

        with torch.no_grad():
            for sample in tqdm(train_dataloader):
                # sample = [s.get_subsequence(60) for s in sample]
                processed_sample = BatchMotionSequence(sample)
                prediction = train.predict_motion(self.config, model_dict, processed_sample)
                predicted_motion = prediction["prediction"]
                for k, loss_fn in loss_dict.items():
                    losses[k].append(loss_fn(prediction=predicted_motion, groundtruth=processed_sample).item())

        for key, loss in losses.items():
            logging.info("Training performance --- %10s: %.5f", key, sum(loss)/len(loss))


    def model_stats(self, epoch = -1, repetitions=300):
        model_dict = self.get_model_dict(epoch)

        #get number of parameters
        num_param_motion_model = sum(p.numel() for p in model_dict["motion_model"].parameters() if p.requires_grad)
        logging.info("Motion model has %d parameters", num_param_motion_model)

        #dummy data/codes
        dummy_sequence = torch.zeros(1, model_dict["sequence_code"].code_dim)
        dummy_action = torch.zeros(1, model_dict["action_code"].code_dim)
        dummy_code = torch.cat([dummy_sequence, dummy_action], 1).to(self.test_config["device"])
        dummy_time_steps = [torch.arange(60, device=self.test_config["device"])]

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                self.generate_motions(dummy_code, dummy_time_steps, None, model_dict)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        stats = torch.tensor(timings[1:])
        logging.info("Model inference time: %.6f ms +- %.6f ms", stats.mean(), stats.std())

    def get_model_dict(self):
        if self.model_dict is None:
            if self.test_config["epoch"] == "latest.pt":
                path_checkpoint = self.path_results.joinpath("checkpoints").joinpath(self.test_config["epoch"])
                epoch = -1
            else:
                path_checkpoint = self.path_results
                epoch = self.test_config["epoch"]
            if self.sample_ids is None:
                try:
                    model_dict, _ = self.trainer.init_model(self.chunked_filelist, self.chunked_action_labels, device=self.test_config["device"], path_ckpt=path_checkpoint, epoch=epoch)
                except AttributeError:
                    model_dict, _ = self.trainer.init_model(self.filelist, self.action_labels, device=self.test_config["device"], path_ckpt=path_checkpoint, epoch=epoch)
            else:
                model_dict, _ = self.trainer.init_model(self.sample_ids, self.action_labels, device=self.test_config["device"], path_ckpt=path_checkpoint, epoch=epoch)
        else:
            model_dict = self.model_dict
        return model_dict

    def generate_save_motions(self, epoch = -1, fixed_root=False):
        path_visual = self.path_eval.joinpath("anim")
        
        if not path_visual.exists():
            path_visual.mkdir()

        path_visual = path_visual.joinpath(self.config.time)
        if not path_visual.exists():
            path_visual.mkdir()

        model_dict = self.get_model_dict()
        model_dict["action_code"].eval()
        model_dict["motion_model"].eval()
        dist_dict = self.fit_distribution(model_dict)

        with torch.no_grad():
            for label, distrib in dist_dict.items():
                llist = [30, 60, 80, 100, 150]
                for l in llist:
                    pred_length = torch.tensor([l])
                    time_steps = torch.arange(l, device=self.test_config["device"]).unsqueeze(0)
                    codes = distrib.sample_n(pred_length, device=self.test_config["device"])
                    motion = self.generate_motions(codes, time_steps, label, model_dict, fixed_root=fixed_root)
                    vertices = motion.batch_conversion("axis_angle", "vertices", return_packed=False)

                    filename_xyz = f"{self.trainer.K['I2A'][label]}_{l}_xyz.npy"
                    filename_vert = f"{self.trainer.K['I2A'][label]}_{l}_vert.npy"
                    np.save(path_visual.joinpath(filename_xyz), torch.stack(motion["joints"]).cpu().numpy())
                    np.save(path_visual.joinpath(filename_vert), torch.stack(vertices).cpu().numpy())


    def generate_video_compilation(self, num_video_per_action=1, width=1024, height=1024, video=True, fixed_root=False):
        from ImplicitMotion.data.render import get_renderer
        import imageio
        path_visual = self.path_eval.joinpath("anim")
        
        if not path_visual.exists():
            path_visual.mkdir()

        path_visual = path_visual.joinpath(self.config.time)
        if not path_visual.exists():
            path_visual.mkdir()
        
        model_dict = self.get_model_dict()
        model_dict["action_code"].eval()
        model_dict["motion_model"].eval()
        dist_dict = self.fit_distribution(model_dict)
        
        renderer = get_renderer(width, height, self.trainer.SMPL_MODEL.faces)
        background = np.zeros((height, width, 3))
        cam=(0.75, 0.75, 0, 0.10)
        color=[0.11, 0.53, 0.8]
        logging.info("Storing videos for %s at %s", self.test_config["path_results"].split("/")[-1], str(path_visual))
        with torch.no_grad():
            label_videos = {}
            for label in [10, 11, 3, 4, 7]:
                distrib = dist_dict[label]
            # for label, distrib in dist_dict.items():
                vid_list = []
                for i in range(num_video_per_action):
                    vids = {}
                    llist = [150]
                    for l in llist:
                        pred_length = torch.tensor([l])
                        time_steps = torch.arange(l, device=self.test_config["device"]).unsqueeze(0)
                        codes = distrib.sample_n(pred_length, device=self.test_config["device"])

                        motion = self.generate_motions(codes, time_steps, label, model_dict, fixed_root=fixed_root)
                        vertices = motion.batch_conversion("axis_angle", "vertices", return_packed=False)
                        
                        frames = [renderer.render(background, frame.detach().cpu().numpy(), cam, color=color, transparency=not video) for frame in vertices[0]]
                        frames = [frame[:, 200:-200, :] for frame in frames]
                        frames = self.add_annotation_to_frame(frames, str(l) + " time steps")

                        # while len(frames) < 100:
                        #     frames.append(frames[-1])
                        vids[str(l)] = frames
                    vid_list.append(vids)
                label_videos[label] = vid_list
                    # vid_list.append([np.concatenate([l1_frame,l2_frame,l3_frame], 1) for l1_frame, l2_frame, l3_frame in zip(*[vids[str(l)] for l in llist])])
                # label_frames = [f for vid in vid_list for f in vid]
                # label_frames = self.add_annotation_to_frame(label_frames, self.trainer.K["I2A"][label], top=False)
                # label_videos[label] = label_frames

            # full_video = [f for vid in label_videos.values() for f in vid]
            # path_visual_file = path_visual.joinpath("full_gen_example.mp4")
            # writer = imageio.get_writer(path_visual_file, fps=25)
            # for frame in full_video:
            #     writer.append_data(frame)
            # writer.close()

            for label, vid_list in label_videos.items():
                for vids in vid_list:
                        for l, frames in vids.items():
                            path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{l}.png")
                            grid = np.concatenate(frames[::3], axis=1)
                            grid[grid[:, :, -1] <= 127] = 0
                            imageio.imsave(path_visual_file, grid)
            # for i, vert in enumerate(vertices):
            #     if video:
            #         path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{length}.mp4")
            #         writer = imageio.get_writer(path_visual_file, fps=25)
            #         for frame in frames:
            #             writer.append_data(frame)
            #         writer.close()
            #     else:
            #         path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{length}.png")
            #         grid = np.concatenate(frames[::3], axis=1)
            #         grid[grid[:, :, -1] <= 127] = 0
            #         imageio.imsave(path_visual_file, grid)

    def generate_video(self, epoch = -1, num_video_per_action=1, width=1024, height=1024, length=60, video=True, smpl=True, fixed_root=False):
        from ImplicitMotion.data.render import get_renderer
        import imageio
        path_visual = self.path_eval.joinpath("anim")
        
        if not path_visual.exists():
            path_visual.mkdir()

        path_visual = path_visual.joinpath(self.config.time)
        if not path_visual.exists():
            path_visual.mkdir()
        
        model_dict = self.get_model_dict()
        model_dict["action_code"].eval()
        model_dict["motion_model"].eval()
        dist_dict = self.fit_distribution(model_dict)
        pred_length = torch.tensor([length]).repeat(num_video_per_action)
        time_steps = torch.arange(length, device=self.test_config["device"]).repeat(num_video_per_action, 1)
        renderer = get_renderer(width, height, self.trainer.SMPL_MODEL.faces)
        background = np.zeros((height, width, 3))
        cam=(0.75, 0.75, 0, 0.10)
        color=[0.11, 0.53, 0.8]
        logging.info("Storing videos for %s at %s", self.test_config["path_results"].split("/")[-1], str(path_visual))
        with torch.no_grad():
            # for label in [10, 11, 3, 4, 7]:
                # distrib = dist_dict[label]
            for label, distrib in dist_dict.items():

                codes = distrib.sample_n(pred_length, device=self.test_config["device"])

                motion = self.generate_motions(codes, time_steps, label, model_dict, fixed_root=fixed_root)
                if smpl:
                    vertices = motion.batch_conversion("axis_angle", "vertices", return_packed=False)

                    for i, vert in enumerate(vertices):
                        frames = [renderer.render(background, frame.detach().cpu().numpy(), cam, color=color, transparency=not video) for frame in vert]

                        #trim
                        frames = [frame[:, 200:-200, :] for frame in frames]

                        if video:
                            path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{length}.mp4")
                            writer = imageio.get_writer(path_visual_file, fps=30)
                            for frame in frames:
                                writer.append_data(frame)
                            writer.close()
                        else:
                            path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{length}.png")
                            grid = np.concatenate(frames[::3], axis=1)
                            grid[grid[:, :, -1] <= 127] = 0
                            imageio.imsave(path_visual_file, grid)
                else:
                    joints = motion.forward_kinematics()
                    for i, seq in enumerate(joints):
                        path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_l{length}.gif")
                        ani = self.trainer.get_animator(1,1, limits=[-1, 1])
                        ani.add_sequence(seq.cpu().detach().numpy(), title=self.trainer.K["I2A"][label])
                        ani.save(path_visual_file)

    def visualize_reconstruction(self):
        from ImplicitMotion.data.render import get_renderer
        import imageio

        path_visual = self.path_eval.joinpath("anim")
        
        if not path_visual.exists():
            path_visual.mkdir()

        path_visual = path_visual.joinpath(self.config.time)
        if not path_visual.exists():
            path_visual.mkdir()

        model_dict = self.get_model_dict()
        model_dict["action_code"].eval()
        model_dict["motion_model"].eval()

        dataset = self.trainer.init_dataloader(device=self.test_config["device"], split="train").dataset

        full_summary = []
        for label, label_filelist in dataset.action_dict.items():
            if label == 'all':
                continue
            label_filelist = set([f for f in label_filelist if f[0] != 'M' and f in self.filelist]) # remove augmented codes

            # Get names of files that were chunked
            try:
                chunked_label_filelist = set(["_".join(f.split("_")[:5]) for f in self.chunked_action_dict[label] if len(f.split("_")) == 7]) # list of samples that were chunked
                chunked_files_dict = {file: {"id": [f for f in self.chunked_action_dict[label] if file in f and len(f.split("_")) == 7]} for file in chunked_label_filelist} # map sample id to its chunks
                label_filelist = list(label_filelist - chunked_label_filelist) # remove chunked files from filelist
                chunked_label_filelist = list(chunked_label_filelist) 
            except AttributeError:
                pass

            label = int(label)

            picked_files = np.random.choice(list(label_filelist), self.test_config["num_videos"])

            sequence_options = {"id": picked_files}
            codes = model_dict["sequence_code"](sequence_options)["code"]

            groundtruth_motions = BatchMotionSequence([dataset[dataset.filelist.index(f)] for f in sequence_options["id"]])

            time_steps = [torch.arange(60, device=self.test_config["device"]) for motion in groundtruth_motions.motion_sequences]

            model_dict["motion_model"].train()
            generated_motions_jitter = self.generate_motions(codes, time_steps, label, model_dict)

            model_dict["motion_model"].eval()
            generated_motions = self.generate_motions(codes, time_steps, label, model_dict)

            width=1024
            height=1024
            renderer = get_renderer(width, height, self.trainer.SMPL_MODEL.faces)
            background = np.zeros((height, width, 3))
            cam=(0.75, 0.75, 0, 0.10)
            color=[0.11, 0.53, 0.8]
            vertices_gt = groundtruth_motions.batch_conversion("axis_angle", "vertices", return_packed=False)
            vertices_pd = generated_motions.batch_conversion("axis_angle", "vertices", return_packed=False)
            vertices_pd_jitter = generated_motions_jitter.batch_conversion("axis_angle", "vertices", return_packed=False)
            label_frames = []
            for i, (vert_gt, vert_pd, vert_jitter) in enumerate(zip(vertices_gt, vertices_pd, vertices_pd_jitter)):
                try:
                    frames_gt = [renderer.render(background, frame.detach().cpu().numpy(), cam, color=color, transparency=False) for frame in vert_gt[:60]]
                    frames_pd = [renderer.render(background, frame.detach().cpu().numpy(), cam, color=color, transparency=False) for frame in vert_pd[:60]]
                    frames_pd_jitter = [renderer.render(background, frame.detach().cpu().numpy(), cam, color=color, transparency=False) for frame in vert_jitter[:60]]
                except Exception:
                    continue

                # add label
                frames_gt = self.add_annotation_to_frame(frames_gt, "groundtruth")
                frames_pd = self.add_annotation_to_frame(frames_pd, "without bug")
                frames_pd_jitter = self.add_annotation_to_frame(frames_pd_jitter, "with bug")

                frames = [np.concatenate([gt_f[:, 200:-200], pd_f[:, 200:-200], pd_f_jitter[:, 200:-200]], 1) for gt_f, pd_f, pd_f_jitter in zip(frames_gt, frames_pd, frames_pd_jitter)]
                frames = self.add_annotation_to_frame(frames, self.trainer.K["I2A"][label], top=False)
                path_visual_file = path_visual.joinpath(self.trainer.K["I2A"][label] + f"_{i}_recon.mp4")
                writer = imageio.get_writer(path_visual_file, fps=30)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

                label_frames.append(frames)
            full_summary.append(label_frames)
        full_summary_frame = [frame for label_frames in full_summary for frames in label_frames for frame in frames]
        path_visual_file = path_visual.joinpath("full_recon.mp4")
        writer = imageio.get_writer(path_visual_file, fps=25)
        for frame in full_summary_frame:
            writer.append_data(frame)
        writer.close()

    def add_annotation_to_frame(self, frames, txt, top=True):
        font = ImageFont.truetype("arial.ttf", 40)
        annotated_frames = []
        x = frames[0].shape[1]/2 - 40
        y = 10 if top else frames[0].shape[0] - 200
        for frame in frames:
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.text((x,y), txt, (0,0,0),font=font)
            annotated_frames.append(np.array(im))
        return annotated_frames

    def evaluate(self, replications = 20, test_dataset = None):
        model_dict = self.get_model_dict()
        model_dict["action_code"].eval()
        model_dict["motion_model"].eval()

        if test_dataset is None:
            groundtruth_dataset = self.train_dataset
        else:
            groundtruth_dataset = test_dataset

        groundtruth_dataloader = torch.utils.data.DataLoader(
            groundtruth_dataset,
            batch_size=512,
            # sampler=torch.utils.data.RandomSampler(groundtruth_dataset, replacement=True, num_samples=test_config["num_samples"]),
            shuffle=True,
            collate_fn=groundtruth_dataset.collate_fn
        )

        stats = {
            "acc": [],
            "fid": [],
            "div": [],
            "mul": [],
            "dis": []
        }

        stats_real = {
            "acc": [],
            "fid": [],
            "div": [],
            "mul": [],
            "dis": []
        }

        dist_dict = self.fit_distribution(model_dict)
        with torch.no_grad():
            for i in range(replications):
                accuracy_real, confusion, features_real, labels_real = self.calculate_accuracy(groundtruth_dataloader)
                _, _, features_real2, labels_real2 = self.calculate_accuracy(groundtruth_dataloader)
                fid_real = a2m.calculate_frechet_distance(
                    features_real.mean(axis=0),
                    test.cov(features_real, rowvar=False),
                    features_real2.mean(axis=0),
                    test.cov(features_real2, rowvar=False))

                diversity_real = calculate_diversity(features_real)

                multimodality_real = calculate_multimodality(groupby(features_real, labels_real))
                effective_distance_real = calculate_effective_distance(features_real, features_real2)

                motion_dict = {}
                for label, distrib in dist_dict.items():
                    if self.test_config["uniform_action_sampling"]:
                        num_samples = self.test_config["num_samples"]
                    else:
                        num_samples = len(list(filter(lambda x: x[0]!='M', self.train_dataset.action_dict[str(label)])))
                        # num_samples = len(self.dataset.action_dict[str(label)])

                    codes, time_steps = self.sample_latent(distrib, num_samples, label)

                    motion_dict[label] = self.generate_motions(codes, time_steps, label, model_dict)

                eucl_motion_loader = torch.utils.data.DataLoader(
                    list(itertools.chain.from_iterable([batch.motion_sequences for batch in motion_dict.values()])),
                    batch_size=512,
                    collate_fn=groundtruth_dataset.collate_fn)

                accuracy, confusion, features_generated, labels_generated = self.calculate_accuracy(eucl_motion_loader)

                fid = a2m.calculate_frechet_distance(
                    features_generated.mean(axis=0),
                    test.cov(features_generated, rowvar=False),
                    features_real.mean(axis=0),
                    test.cov(features_real, rowvar=False))
                diversity = calculate_diversity(features_generated)

                multimodality = calculate_multimodality(groupby(features_generated, labels_generated))

                effective_distance = calculate_effective_distance(features_generated, features_real)

                stats["acc"].append(accuracy)
                stats["fid"].append(fid)
                stats["div"].append(diversity)
                stats["mul"].append(multimodality)
                stats["dis"].append(effective_distance)
                stats_real["acc"].append(accuracy_real)
                stats_real["fid"].append(fid_real)
                stats_real["div"].append(diversity_real)
                stats_real["mul"].append(multimodality_real)
                stats_real["dis"].append(effective_distance_real)
                ### Logging
                confusion_msg = f"\n---------------- Confusion matrix {i} -----------------\n"
                for j, row in enumerate(confusion):
                    confusion_msg += " ".join([f"{col.item():3d}" for col in row]) + f" | {(row[j]/row.sum()).item():.3f} | " + self.trainer.K["I2A"][j] +"\n"
                logging.info(confusion_msg)

                logging.info(f"Real {i} " + " | ".join([f"{key} : {val[-1]:.3f}" for key, val in stats_real.items()]))
                logging.info(f"Experiment {i} " + " | ".join([f"{key} : {val[-1]:.3f}" for key, val in stats.items()]))

        logging.info(
            f"Summary Real " + " | ".join([f"{key} : {sum(val)/len(val):.3f} +- {(1.96*torch.std(torch.tensor(val))/math.sqrt(replications)).item():.3f}" for key, val in stats_real.items()]))
        logging.info(
            f"Summary " + " | ".join([f"{key} : {sum(val)/len(val):.3f} +- {(1.96*torch.std(torch.tensor(val))/math.sqrt(replications)).item():.3f}" for key, val in stats.items()]))

        return stats

    def generate_motions(self, codes, time_steps, label, model_dict, fixed_root=False):
        if self.config.positional_embedding_type != "fixed":
            sequence_code = codes[:, :model_dict["sequence_code"].code_dim]
            time_code = codes[:, model_dict["sequence_code"].code_dim:]
        else:
            sequence_code = codes[:, :model_dict["sequence_code"].code_dim]
            action_code = codes[:, model_dict["sequence_code"].code_dim:]
            time_code = None

        if not action_code.numel():
            try:
                action_code = model_dict["action_code"]({"label": [str(label)] * len(codes)})["code"]
            except KeyError:
                action_code = torch.empty(0, device=self.test_config["device"])

        # Predict time sequences
        time_sequences = model_dict["time_function"].predict(time_code, time_steps)

        # Predict motion (lie)
        if self.config.action_code_additive:
            label_code = sequence_code + action_code
        else:
            label_code = torch.cat([sequence_code, action_code],1)

        prediction = model_dict["motion_model"](time_sequences, label_code)
        try:
            if self.config.recon_loss_type == "joint":
                prediction["joints"] = prediction.forward_kinematics()
            else:
                prediction["joints"]
            if fixed_root:
                prediction["root"] = [pred-pred[:, 0, None, :] for pred in prediction["root"]]
        except NotImplementedError:
            prediction = BatchMotionSequence([MotionSequence({"joints": seq}, str(label), 0) for seq in prediction.forward_kinematics()])
        prediction["label"] = str(label)
        return prediction

    def sample_latent(self, latent_dist, num_samples, label, match_freq=True):
        if self.test_config["variable_length_testing"]:
            if match_freq:
                sample_list = self.train_dataset.action_dict[str(label)]
                pred_lengths = torch.tensor([len(self.train_dataset[self.train_dataset.filelist.index(s)]) for s in sample_list])
                if self.config.batch_subsampling != "fixed_length":
                    time_steps = [torch.arange(1, pdl+1, device=self.test_config["device"], dtype=torch.float32) for pdl in pred_lengths]
                else:
                    time_steps = [self.train_dataset[self.train_dataset.filelist.index(s)].get_subsequence(self.test_config["fixed_pred_length"]).time_idx for s in sample_list]
            else:
                pred_lengths, time_steps = self.sample_time_steps(num_samples, latent_dist.max_len, min_len = latent_dist.min_len)

            if self.test_config["upper_limit_testing"]:
                codes, new_lengths = latent_dist.sample_n(pred_lengths, device=self.test_config["device"])
                pred_lengths = new_lengths
                time_steps = [torch.arange(1, pdl+1, device=self.test_config["device"], dtype=torch.float32) for pdl in pred_lengths]
            elif self.test_config["length_conditional"]:
                codes = latent_dist.sample_n(pred_lengths, device=self.test_config["device"])
            else:
                codes = latent_dist.sample_n(len(pred_lengths), device=self.test_config["device"])

        else:
            codes = latent_dist.sample_n(num_samples, device=self.test_config["device"])
            pred_lengths, time_steps = self.sample_time_steps(num_samples, self.test_config["fixed_pred_length"], min_len = -1)

        return codes, time_steps

    def sample_time_steps(self, num_samples, max_len, min_len=-1):
        if min_len != -1:
            pred_lengths = torch.randint(low=min_len, high=max_len, size=(num_samples,))
            time_steps = [torch.arange(1, pdl+1, device=self.test_config["device"], dtype=torch.float32) for pdl in pred_lengths]
        else:
            pred_lengths = torch.tensor(self.test_config["fixed_pred_length"]).expand(num_samples)
            time_steps = torch.arange(1, max_len+1, device=self.test_config["device"], dtype=torch.float32).unsqueeze(0).expand(num_samples, -1)
        return pred_lengths, time_steps

    def fit_distribution(self, model_dict):
        ### Fit distribution to latent space
        dist_dict = {}

        if not model_dict["sequence_code"].opts.variational:
            self.test_config["num_variational_samples"] = 1
        for label, label_filelist in self.action_dict.items():
            if label == 'all':
                continue
            label_filelist = set([f for f in label_filelist if f[0] != 'M' and f in self.filelist]) # remove augmented codes

            # Get names of files that were chunked
            try:
                chunked_label_filelist = set(["_".join(f.split("_")[:5]) for f in self.chunked_action_dict[label] if len(f.split("_")) == 7]) # list of samples that were chunked
                chunked_files_dict = {file: {"id": [f for f in self.chunked_action_dict[label] if file in f and len(f.split("_")) == 7]} for file in chunked_label_filelist} # map sample id to its chunks
                label_filelist = list(label_filelist - chunked_label_filelist) # remove chunked files from filelist
                chunked_label_filelist = list(chunked_label_filelist) 
            except AttributeError:
                pass

            label = int(label)

            sequence_options = {"id": label_filelist}

            if self.test_config["num_variational_samples"] == 1:
                model_dict["sequence_code"].eval()
                model_dict["time_function"].eval()

            label_sequence_code = torch.cat([model_dict["sequence_code"](sequence_options)["code"] for _ in range(self.test_config["num_variational_samples"])])

            try:
                chunked_codes = [torch.stack([model_dict["sequence_code"](chunked_files_dict[file])["code"][0] for _ in range(self.test_config["num_variational_samples"])]) for file in chunked_label_filelist]
                # chunked_codes = [torch.stack([torch.zeros(model_dict["sequence_code"].code_dim, device=label_sequence_code.device) for _ in range(self.test_config["num_variational_samples"])]) for file in chunked_label_filelist]
                label_sequence_code = torch.cat([label_sequence_code, torch.cat(chunked_codes)])
            except NameError:
                pass

            try:
                label_time_code = torch.cat([model_dict["time_function"].time_codes(sequence_options)["code"] for _ in range(self.test_config["num_variational_samples"])])
                codes = torch.cat([label_sequence_code, label_time_code], 1)
            except (AttributeError, KeyError):
                codes= label_sequence_code

            try:
                label_action_code = torch.cat([model_dict["action_code"](sequence_options)["code"] for _ in range(self.test_config["num_variational_samples"])])
                codes = torch.cat([codes, label_action_code], 1)
            except KeyError:
                pass

            if self.test_config["variable_length_testing"]:
                length_list = [len(self.train_dataset.data[file]["root"]) for _ in range(self.test_config["num_variational_samples"]) for file in label_filelist]
                try:
                    chunked_length_list =  [len(self.train_dataset.data[file]["root"]) for _ in range(self.test_config["num_variational_samples"]) for file in chunked_label_filelist]
                    length_list += chunked_length_list
                except NameError:
                    pass

                if self.test_config["upper_limit_testing"]:
                    dist_dict[label] = dist.VariableLengthRandomSampler()
                    dist_dict[label].fit(codes, length_list)
                elif self.test_config["length_conditional"]:
                    dist_dict[label] = dist.LengthConditionalDistribution(gaussian_mixture=self.test_config["gaussian_mixture_components"])
                    dist_dict[label].fit(codes, length_list)
                    logging.info("%s: %s", label, dist_dict[label])
                else:
                    dist_dict[label] = dist.GaussianMixtureTorch(self.test_config["gaussian_mixture_components"], "full", len_list=length_list)
                    dist_dict[label].fit(codes)
                    logging.info("%s: %s", label, dist_dict[label])
            else:
                if self.test_config["upper_limit_testing"]:
                    dist_dict[label] = dist.RandomSampler()
                    dist_dict[label].fit(codes)
                else:
                    dist_dict[label] = dist.GaussianMixtureTorch(self.test_config["gaussian_mixture_components"], "full")
                    dist_dict[label].fit(codes)
                    logging.info("%s: %s", label, dist_dict[label])
        return dist_dict

    def calculate_accuracy(self, motion_loader):
        confusion = torch.zeros(self.config.dataset_opts.num_labels, self.config.dataset_opts.num_labels, dtype=torch.long)

        activations = []
        labels = []
        for batch in motion_loader:
            batch_prob, activation = self.classifier(batch)
            batch_pred = batch_prob.max(dim=1).indices
            activations.append(activation)

            labels.append(torch.tensor([int(seq.label) for seq in batch]))

            for label, pred in zip(labels[-1], batch_pred):
                confusion[int(label)][pred] += 1

        accuracy = (confusion.trace()/confusion.sum()).item()
        return accuracy, confusion, torch.cat(activations), torch.cat(labels)

def groupby(activations, labels):
    return {i:torch.stack([act for act, ii in zip(activations, labels) if ii == i]) for i in labels.unique()}

def calculate_effective_distance(features_generated, features_real):
    distance = torch.stack([(features_real - feat.unsqueeze(0)).norm(2, -1).min() for feat in features_generated]).mean().item()
    return distance

def calculate_diversity(activations, diversity_times=200):
    num_motions = len(activations)
    first_indices = torch.randint(0, high=num_motions, size=(diversity_times,))
    second_indices = torch.randint(0, high=num_motions, size=(diversity_times,))

    return torch.norm(activations[first_indices] - activations[second_indices], 2, -1).mean().item()

def calculate_multimodality(activations_dict, multimodality_times=20):
    return sum([calculate_diversity(act, diversity_times=multimodality_times) for key, act in activations_dict.items()]) / len(activations_dict)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_results", help="Location of the experiment")
    parser.add_argument("--epoch", default="latest.pt")
    parser.add_argument("--test_split", action="store_true", help="Test split only on UESTC")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--path_classifier", default="./ImplicitMotion/model_file", help="Path to the folder that contains classifiers. Classifier name is defined in trainer")
    parser.add_argument("--uniform_action_sampling", action="store_true", help="Uniform sampling was used by Action2Motion")
    parser.add_argument("--num_samples", default=300, help="Used when running with uniform_action_sampling")

    parser.add_argument("--gaussian_mixture_components", default=15, type=int, help="number of components per GMM")

    parser.add_argument("--variable_length_testing", action="store_true")
    parser.add_argument("--fixed_pred_length", default=60)
    parser.add_argument("--upper_limit_testing", action="store_true", help="Pick optimized codes only")
    parser.add_argument("--length_conditional", action="store_false", help="Sequence-length conditional latent space")
    parser.add_argument("--num_variational_samples", default=50)

    parser.add_argument("--metrics", action="store_true", help="Compute metrics")
    parser.add_argument("--video", action="store_true", help="Generate videos")
    parser.add_argument("--num_videos", default=3, type=int, help="Number of videos to generate per action")
    parser.add_argument("--video_length", default=60, type=int, help="Length of generated videos")

    return parser.parse_args()

if __name__ == "__main__":

    opts = parse()
    test_config = opts.__dict__

    eval = EvaluationMetrics(test_config)
    eval.generate_video_compilation()
    # eval.visualize_reconstruction()

    if opts.metrics:
        if opts.test_split:
            dataloader = eval.trainer.init_dataloader(device=test_config["device"], split="test")
            eval.evaluate(epoch=test_config["epoch"], test_dataset=dataloader.dataset)
        else:
            eval.evaluate()

    if opts.video:
        eval.generate_video(num_video_per_action=opts.num_videos, length=opts.video_length)

    # eval.model_stats(epoch=test_config["epoch"])
    # eval.evaluate_reconstruction(epoch=test_config["epoch"])

