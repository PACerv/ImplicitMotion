import copy
import itertools

import re
import joblib

import numpy as np

import torch
from smplx import SMPL

from ImplicitMotion.data.motion_sequences import MotionSequence
import ImplicitMotion.data.conversions as T
import ImplicitMotion.data.CONSTANTS as K

EPS = 1e-5

def get_dataset_opts(dataset_name):
    if dataset_name == "Human":
        return HumanAct12.opts
    elif dataset_name == "NTU13":
        return NTUVIBE13.opts
    elif dataset_name == "UESTC":
        return UESTC.opts
    else:
        raise ValueError("Dataset %s unknown", dataset_name)

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, path, fix_root=True, device="cpu", split=None,**kwargs):
        self.base_path = path
        self.filelist = {}

        self.device = device
        self.fix_root = fix_root
        fileargs = self.get_filelist(path, split=split, **kwargs) 

        self.filelist, self.action_labels, self.action_map, self.action_dict = fileargs
        self.data = self.load_data(device=device, split=split)
        self.motion_sequences = self.construct_motion_sequences()

    def load_data(self, device="cpu", split=None):
        data =  {filename: self.load_sample_by_filename(filename, device=device) for filename in self.filelist}

        for file_name, sample in data.items():
            if self.fix_root:
                # move the root of all poses at t=0 to origin
                sample = self.root_to_origin(sample)
            data[file_name] = sample

        return data

    def construct_motion_sequences(self):
        return [MotionSequence(self.data[sample_id], sample_label, sample_id) for sample_label, sample_id in zip(self.action_labels, self.filelist)]


    @classmethod
    def recover_augmented_sample_description(cls, filename):
        fields = filename.split("_")
        return {
            "action_label": fields[1],
            "start_sequence_idx": int(fields[2]),
            "start_range": [int(i) for i in fields[3:5]],
            "end_sequence_idx": int(fields[5]),
            "end_range": [int(i) for i in fields[6:]]
        }

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, i):
        return self.motion_sequences[i]

    @classmethod
    def collate_fn(cls, samples):
        return samples

class HumanAct12(MotionDataset):
    opts = {
        "num_joints": 24,
        "num_labels": 12
    }
    _UNFILTERED_FILELIST = None

    def __init__(self, *args, use_SMPL, device, path_smpl=None, **kwargs):
        self.use_SMPL = use_SMPL
        if self.use_SMPL:
            self.SMPL_data = joblib.load(args[0].joinpath('HumanAct12Poses').joinpath("humanact12poses.pkl"))
            self.SMPL_MODEL = SMPL(path_smpl).to(device)
            MotionSequence.load_smpl_model(self.SMPL_MODEL)
        if self._UNFILTERED_FILELIST is None:
            self.get_full_filelist(args[0])
        MotionSequence.load_kinematic_tree(K.HUMAN_bone_hierarchy)
        MotionSequence.load_default_skeleton(K.HUMAN_default_bones)
        super().__init__(*args, device=device,**kwargs)

    def root_to_origin(self, sample):
        sample["joints"] = sample["joints"] - sample["root"][0, None, :, :]
        sample["root"] = sample["root"] - sample["root"][0, None, :, :]
        return sample

    def load_sample_by_filename(self, filename, device="cpu"):
        if self.use_SMPL:
            i = self.get_full_filelist(self.base_path).index(filename)
            joints = torch.from_numpy(self.SMPL_data["joints3D"][i]).float().to(device)
            sample = {
                "axis_angle": torch.from_numpy(self.SMPL_data["poses"][i]).float().reshape(-1, 24,3).to(device),
                "joints": joints[:, 1:, :].contiguous(),
                "root": joints[:, 0, None, :].contiguous()
            }
            return sample
        else:
            filepath = self.base_path.joinpath(filename + ".npy")
            data = torch.from_numpy(np.load(filepath)).type(torch.float32)
            return {
                "joints": data[:, 1:, :].contiguous().to(device),
                "root": data[:, 0, None, :].contiguous().to(device)
                }

    @classmethod
    def get_action_from_filename(cls, filename):
        return str(int(filename[-4:-2])-1)

    @classmethod
    def get_full_filelist(cls, path):
        filelist = [f.stem for f in path.iterdir() if f.suffix == ".npy"]
        filelist.sort()
        cls._UNFILTERED_FILELIST = filelist
        return filelist

    @classmethod
    def get_filelist(cls, path, split=None, **kwargs):
        if cls._UNFILTERED_FILELIST is None:
            filelist = cls.get_full_filelist(path)
        else:
            filelist = copy.deepcopy(cls._UNFILTERED_FILELIST)

        sorted_filelist = sorted(filelist, key=cls.get_action_from_filename)
        action_dict = {}
        # groupby only considers consequtive groups so sort first
        for k, v in itertools.groupby(sorted_filelist, key=cls.get_action_from_filename):
            action_dict[k] = list(v)

        action_map = {f:k for k,v in action_dict.items() for f in v}
        # filelist = list(action_map.keys())
        action_labels = [action_map[f] for f in filelist]
        action_dict['all'] = filelist
        return filelist, action_labels, action_map, action_dict

#########################################################################
#########################################################################

class NTUVIBE13(MotionDataset):
    opts = {
        "num_joints": 18,
        "num_labels": 13
    }

    def __init__(self, *args, **kwargs):
        MotionSequence.load_kinematic_tree(K.NTU_VIBE_bone_hierarchy)
        MotionSequence.load_default_skeleton(K.NTU_VIBE_default_bones)
        super().__init__(*args, **kwargs)

    def load_sample_by_filename(self, filename, device="cpu"):
        kinect_vibe_extract_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]
        joints = torch.from_numpy(joblib.load(self.base_path.joinpath(f"{filename}.pkl"))[1]['joints3d'][:, kinect_vibe_extract_joints, :]).to(device)

        return {"joints": joints[:, 1:, :], "root": joints[:, 0, None, :]}

    def root_to_origin(self, sample):
        sample["joints"] = sample["joints"] - sample["root"][0, None, :]
        sample["root"] = sample["root"] - sample["root"][0, None, :, :]
        return sample

    @classmethod
    def get_filelist(cls, path, split=None, **kwargs):
        filelist = [p.stem for p in path.iterdir() if 1 in joblib.load(p)]
        
        if split == "cross_setup_train":
            filelist = list(filter(lambda p: int(re.findall("(?<=S)\d+", p)[0]) % 2 == 0,filelist))
        elif split == "cross_setup_test":
            filelist = list(filter(lambda p: int(re.findall("(?<=S)\d+", p)[0]) % 2 == 1,filelist))
        elif split == "cross_subject_train":
            filelist = list(filter(lambda p: int(re.findall("(?<=P)\d+", p)[0]) in K.NTU_train_subjects, filelist))
        elif split == "cross_subject_test":
            filelist = list(filter(lambda p: int(re.findall("(?<=P)\d+", p)[0]) not in K.NTU_train_subjects, filelist))
        elif split == "full":
            pass
        else:
            raise ValueError(f"split {split} not implemented")

        sorted_filelist = sorted(filelist, key=lambda x: str(int(x[-3:])))
        action_dict = {}
        # groupby only considers consequtive groups so sort first
        for k, v in itertools.groupby(sorted_filelist, key=lambda x: str(int(x[-3:])-1)):
            action_dict[k] = list(v)

        sorted_labels = [5, 6, 7, 8, 21, 22, 23, 37, 79, 92, 98, 99, 101]
        map_new_labels = {i:j for i,j in zip(sorted_labels, list(range(len(sorted_labels))))}

        new_action_dict = {}
        for k,v in action_dict.items():
            new_action_dict[str(map_new_labels[int(k)])] = v
        action_dict = new_action_dict

        action_map = {f: k for k,v in action_dict.items() for f in v}

        filelist = list(action_map.keys())
        action_labels = list(action_map.values())
        return filelist, action_labels, action_map, action_dict

#########################################################################
#########################################################################

class UESTC(MotionDataset):
    opts = {
        "num_joints": 24,
        "num_labels": 40
    }
    TRAIN_IDs = [
        1, 2, 6, 12, 13, 16, 21, 24, 28, 29,
        30, 31, 33, 35, 39, 41, 42, 45, 47, 50,
        52, 54, 55, 57, 59, 61, 63, 64, 67, 69,
        70, 71, 73, 77, 81, 84, 86, 87, 88, 90,
        91, 93, 96, 99 ,102, 103, 104, 107, 108, 112, 113]

    _UNFILTERED_FILELIST = None

    def __init__(self, *args, use_SMPL, device, path_smpl=None, **kwargs):
        self.use_SMPL = use_SMPL
        
        self.SMPL_MODEL = SMPL(path_smpl).eval().to(device)
        MotionSequence.load_smpl_model(self.SMPL_MODEL)
        MotionSequence.load_kinematic_tree(K.UESTC_bone_hierarchy)
        MotionSequence.load_default_skeleton(K.UESTC_default_bones)
        self.transl = joblib.load(args[0].joinpath("globtrans_usez.pkl"))
        self.motion_data = joblib.load(args[0].joinpath("vibe_cache_refined.pkl"))
        self.rotations = {key: self.get_view_rotation(key, device) for key in [0, 1, 2, 3, 4, 5, 6, 7]}

        if self._UNFILTERED_FILELIST is None:
            self.get_full_filelist(args[0])
        
        super().__init__(*args, device=device, **kwargs)

    @classmethod
    def get_full_filelist(cls, path):
        path_names = path.joinpath("info").joinpath("names.txt")
        with path_names.open() as f:
            names = [l[:-4] for l in f.read().splitlines()]
        cls._UNFILTERED_FILELIST = names
        return names

    def root_to_origin(self, sample):
        sample["joints"] = sample["joints"] - sample["root"][0, None, :, :]
        sample["root"] = sample["root"] - sample["root"][0, None, :, :]
        return sample

    def get_view_rotation(self, view, device):
        theta = - view * np.pi/4
        axis = torch.tensor([0, 1, 0], dtype=torch.float)
        axisangle = theta*axis
        matrix = T.axis_angle_to_matrix(axisangle).to(device)
        return matrix

    def load_sample_by_filename(self, filename, device="cpu"):
        info = filename.split('_')
        if len(info) == 7:
            orig_filename = "_".join(info[:5])
            chunk_id = int(info[5][1:])
            num_chunks = int(info[6][1:])
        else:
            orig_filename = filename

        idx = self._UNFILTERED_FILELIST.index(orig_filename)
        get_view = lambda x: int(x.split('_')[1][1:])
        get_side = lambda x: int(x.split('_')[3][1:])

        if len(info) == 7:
            full_length = len(self.motion_data["joints3d"][idx])
            step = int(full_length/num_chunks)
            from_idx = chunk_id * step

            if chunk_id + 1 == num_chunks:
                to_idx = -1 # last chunk takes rest
            else:
                to_idx = (chunk_id+1) * step
        else:
            from_idx = 0
            to_idx = -1

        joints3D = torch.from_numpy(self.motion_data["joints3d"][idx][from_idx:to_idx, :24, :]).float().to(device)
        transl = torch.from_numpy(self.transl[idx][from_idx:to_idx, None, :]).float().to(device)
        axis_angle = torch.from_numpy(self.motion_data["pose"][idx][from_idx:to_idx]).reshape(-1, 24, 3).float().to(device)

        if get_side(filename) == 1:
            pass
        else:
            rotation = self.rotations[get_view(filename)]
            global_rot = T.axis_angle_to_matrix(axis_angle[:, 0])
            axis_angle[:, 0, :] = T.quaternion_to_axis_angle(T.matrix_to_quaternion(rotation @ global_rot))
            joints3D = joints3D @ rotation.T
            transl = transl @ rotation.T

        joints3D = joints3D + transl

        sample = {
            "axis_angle": axis_angle,
            "joints" : joints3D[:, 1:],
            "root": joints3D[:, 0, None]
        }
        return sample

    @classmethod
    def get_filelist(cls, path, *args, chunk_limit, approx_chunk_size, split=None, **kwargs):
        path_names = path.joinpath("info").joinpath("names.txt")
        with path_names.open() as f:
            names = [l[:-4] for l in f.read().splitlines()]

        names, remain_idx = cls.filter_filenames(names, split=split)

        if chunk_limit != -1:
            ## Split sequences into chunks
            path_lengths = path.joinpath("info").joinpath("num_frames.txt")
            with path_lengths.open() as f:
                lengths = [int(l) for l in f.read().splitlines()]

            lengths = np.array([lengths[i] for i in remain_idx])

            # sequences larger than chunk_limit are split into segments
            chunk_idx = np.where(lengths>chunk_limit)[0]
            num_chunks = np.array([np.round(lengths[idx]/approx_chunk_size).astype(np.int) for idx in chunk_idx])

            for i, nc in zip(chunk_idx, num_chunks):
                names += [names[i] + f"_S{j}_X{nc}" for j in range(1,nc)]
                names[i] += f"_S0_X{nc}"

        get_action = lambda x: int(x.split("_")[0][1:])

        sorted_names = sorted(names, key=get_action)
        action_dict = {}
        for k, v in itertools.groupby(sorted_names, key=get_action):
            action_dict[str(k)] = list(v)

        action_map = {f:k for k,v in action_dict.items() for f in v}

        action_labels = [action_map[f] for f in names]
        action_dict['all'] = names
        return names, action_labels, action_map, action_dict

    @classmethod
    def filter_filenames(cls, names, split):
        original_names = names.copy()
        is_train_performer = lambda x: int(x.split('_')[2][1:]) in cls.TRAIN_IDs
        is_test_performer = lambda x: int(x.split('_')[2][1:]) not in cls.TRAIN_IDs

        if split == "train":
            names = list(filter(is_train_performer, names))
        elif split == "test":
            names = list(filter(is_test_performer, names))
        else:
            raise ValueError()

        get_view = lambda x: int(x.split('_')[1][1:])
        get_side = lambda x: int(x.split('_')[3][1:])
        is_not_view_eight = lambda x: get_view(x) != 8 or get_side(x) == 1
        names = list(filter(is_not_view_eight, names))
        return names, [original_names.index(n) for n in names]


#########################################################################
#########################################################################

class Partition(torch.utils.data.Dataset):
    def __init__(self, data, indices):
        self.data = [data[idx] for idx in indices]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.data[i]

class DataPartitioner(object):
    def __init__(self, data, num_splits, seed=1234):
        # self.data = data

        self.indices = torch.randperm(len(data))
        self.partitions =  torch.split(self.indices, int(np.ceil(len(data)/num_splits)))

    def use(self, data, partition):
        return Partition(data, self.partitions[partition])


