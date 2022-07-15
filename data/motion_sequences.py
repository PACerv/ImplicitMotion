import torch

import ImplicitMotion.data.conversions as T

class MotionSequence(object):
    _DATA_REPR = ["axis_angle", "rot_6D", "joints", "joints_noroot", "vertices", "vertices_noroot", "root"]
    _SMPL = None
    _KINEMATIC_TREE = None
    _DEFAULT_SKELETON = None
    _BETA = None

    def __init__(self, motion_data, label, id, cache=True, time_idx=None):
        assert all([key in self._DATA_REPR for key in motion_data.keys()]), "Motion data contains unknown data representation"
        # check that all data has same length
        length = set([len(val) for val in motion_data.values()])
        if len(length) > 1:
            raise ValueError("Data with different length provided")

        device = set([val.device for val in motion_data.values()])
        if len(device) > 1:
            raise ValueError("Data on multiple devices")
        self._device = device.pop()

        if time_idx is None:
            self.time_idx = torch.arange(0, length.pop(), device=self._device)
        else:
            self.time_idx = time_idx
        self.motion_data = motion_data
        self.label = label
        self.id = id
        self.cache = cache

    def get_subsample(self, num_samples):
        if num_samples > len(self.time_idx):
            return self.get_subsequence(num_samples)
            # raise ValueError("Subsample larger than original sample")
        idx = torch.randperm(len(self.time_idx))[:num_samples]
        subsample = {k: v[idx] for k,v in self.motion_data.items()}
        return MotionSequence(motion_data=subsample, label=self.label, id=self.id, time_idx=self.time_idx[idx])

    def get_subsequence(self, sequence_length, sample_from_start=False):
        if len(self) < sequence_length:
            gap = sequence_length - len(self)
            subsample = {}
            for k,v in self.motion_data.items():
                last_pose = v[-1].repeat(gap, 1, 1)
                subsample[k] = torch.cat([v, last_pose])
            time_idx = torch.arange(0, sequence_length, device=self.device)
        elif len(self) > sequence_length:
            if sample_from_start:
                start_idx = 0
            else:
                start_idx = torch.randint(0, len(self) - sequence_length, (1,)).item()
            subsample = {k: v.narrow(0, start_idx, sequence_length) for k,v in self.motion_data.items()}
            time_idx = self.time_idx[start_idx:start_idx+sequence_length]
        else:
            return self
        return MotionSequence(motion_data=subsample, label=self.label, id=self.id, time_idx=time_idx)

    def __len__(self):
        length = set([len(val) for val in self.motion_data.values()])
        if len(length) > 1:
            raise ValueError("Data with different length provided")
        return length.pop()

    def __getitem__(self, key):
        assert key in self._DATA_REPR, f"Key {key} unknown."
        if key in self.motion_data:
            return self.motion_data[key]
        else:
            # infer data from other modalities
            for from_key, data in self.motion_data.items():
                try:
                    converted_data = self.conversion_function(from_key, key)(data)
                    if self.cache:
                        self.motion_data[key] = converted_data
                    return converted_data
                except NotImplementedError:
                    continue
            
            # raise appropriate error
            for from_key in self.motion_data.keys():
                self.conversion_function(from_key, key)

    def __setitem__(self, key, newvalue):
        assert key in self._DATA_REPR, f"Key {key} unknown."
        assert all([len(newvalue) == len(val) for val in self.motion_data.values()]), f"Length of data doesn't match. Length {len(newvalue)}"
        self.motion_data[key] = newvalue

    def conversion_function(self, from_repr, to_repr):
        if to_repr == "vertices" and self._SMPL is None:
            raise ValueError("SMPL model not available.")

        if to_repr == "joints" and self._KINEMATIC_TREE is None and self._SMPL is None:
            raise ValueError("Forward kinemetics undefined.")

        if to_repr == "root":
            raise ValueError("Root can't be inferred.")

        if from_repr in ["joints", "vertices"]:
            raise NotImplementedError("Inverse kinematics not implemented.")

        if from_repr == "axis_angle":
            if to_repr == "rot_6D":
                return lambda x: T.axis_angle_to_rotation_6d(x)
            elif to_repr == "joints":
                if self._SMPL is None:
                    raise NotImplementedError("Call forward_kinematics instead.")
                else:
                    if "root" in self.motion_data:
                        return lambda x: getattr(self._SMPL(body_pose=x[:, 1:, :], global_orient=x[:, 0, None, :], transl=self.motion_data["root"].squeeze(), return_verts=False), "joints")[:, :24, :]
                    else:
                        raise ValueError("Root unknown. Call 'joints_noroot' to get joints anyway.")
            elif to_repr == "joints_noroot":
                if self._SMPL is None:
                    raise NotImplementedError("Call foward_kinematics instead")
                else:
                    return lambda x: getattr(self._SMPL(body_pose=x[:, 1:, :], global_orient=x[:, 0, None, :], return_verts=False), "joints")[:, 1:24, :]
            elif to_repr == "vertices":
                if "root" in self.motion_data:
                    return lambda x: getattr(self._SMPL(body_pose=x[:, 1:, :], global_orient=x[:, 0, None, :], transl=self.motion_data["root"].squeeze()), "vertices")
                else:
                    raise ValueError("Root unknown. Call 'vertices_noroot' to get vertices anyway.")
            elif to_repr == "vertices_noroot":
                return lambda x: getattr(self._SMPL(body_pose=x[:, 1:, :], global_orient=x[:, 0, None, :]), "vertices")
            else:
                raise NotImplementedError(f"Can't convert to {to_repr}.")
        elif from_repr == "rot_6D":
            if to_repr == "axis_angle":
                return lambda x: T.rotation_6d_to_axis_angle(x)
            elif to_repr in self._DATA_REPR:
                axis_angle_conversion = self.conversion_function(from_repr, "axis_angle")
                to_repr_conversion = self.conversion_function("axis_angle", to_repr)
                return lambda x: to_repr_conversion(axis_angle_conversion(x))
            else:
                raise NotImplementedError(f"Can't convert to {to_repr}.")
        else:
            raise NotImplementedError(f"Can't convert from {from_repr}.")

    @property
    def device(self):
        #check if all data on same device
        device = set([val.device for val in self.motion_data.values()])
        if len(device) > 1:
            raise ValueError("Data on multiple devices")
        return device.pop()

    @device.setter
    def device(self, new_device):
        for val in self.motion_data.values():
            val.to(new_device)

    def __contains__(self, key):
        return key in self.motion_data

    @classmethod
    def load_smpl_model(cls, smpl_model):
        cls._SMPL = smpl_model
        cls._BETA = torch.zeros([1,smpl_model.num_betas])

    @classmethod
    def load_kinematic_tree(cls, kt):
        cls._KINEMATIC_TREE = kt

    @classmethod
    def load_default_skeleton(cls, skeleton):
        cls._DEFAULT_SKELETON = skeleton

    def forward_kinematics(self, joints=None, bones=None):
        if joints is None and bones is None and self._DEFAULT_SKELETON is None:
            raise ValueError("Either bones or joint locations need to be provided")
        elif joints is not None:
            joint_locs = joints
            bones = None
        elif bones is not None:
            bones = bones
            joint_locs = None
        elif self._DEFAULT_SKELETON is not None:
            bones = self._DEFAULT_SKELETON
            joint_locs = None

        if self._KINEMATIC_TREE is None:
            raise ValueError("Need to define kinematic tree")

        if "axis_angle" in self or "rot_6D" in self:
            if "root" in self.motion_data:
                return T.axis_angle_to_joints(self._KINEMATIC_TREE, self["axis_angle"], root_trajectory=self.motion_data["root"], joint_locs=joint_locs, default_bones=bones)
            else:
                return T.axis_angle_to_joints(self._KINEMATIC_TREE, self["axis_angle"], joint_locs=joint_locs, default_bones=bones)
        else:
            raise ValueError("Not enough information for forward kinematics provided")


class BatchMotionSequence(object):
    def __init__(self, motion_sequences):
        self.motion_sequences = motion_sequences

    def __getitem__(self, key):
        try:
            return [getattr(seq, key) for seq in self.motion_sequences]
        except AttributeError:
            return [seq[key] for seq in self.motion_sequences]

    def __setitem__(self, key, newval):
        if not isinstance(newval, list):
            newval = [newval] * len(self)
        for seq, val in zip(self.motion_sequences, newval):
            try:
                seq[key] = val
            except AssertionError:
                setattr(seq, key, val)

    @property
    def cache(self):
        return any([seq.cache for seq in self.motion_sequences])

    @cache.setter
    def cache(self, new_cache:bool):
        for seq in self.motion_sequences:
            seq.cache = new_cache

    @property
    def device(self):
        #check if all data on same device
        device = set([seq.device for seq in self.motion_sequences])
        if len(device) > 1:
            raise ValueError("Data on multiple devices")
        return device.pop()

    @device.setter
    def device(self, new_device):
        for seq in self.motion_sequences:
            seq.device = new_device

    def __len__(self):
        return len(self.motion_sequences)

    def __contains__(self, key):
        try:
            return all([key in seq for seq in self.motion_sequences])
        except:
            return False

    def unpack(self, data):
        idx_start = 0
        emb_list = []
        for seq in self.motion_sequences:
            emb_list.append(data[idx_start:idx_start+len(seq), :])
            idx_start += len(seq)
        return emb_list

    def batch_conversion(self, from_key, to_key, return_packed=True):
        from_data = {from_key: torch.cat(self[from_key])}
        if to_key in ["joints", "vertices"]:
            if "root" in self:
                from_data["root"] = torch.cat(self["root"])

        packed_sequence = MotionSequence(from_data, "", 0, cache=False)
        if return_packed:
            return packed_sequence[to_key]
        else:
            to_data = self.unpack(packed_sequence[to_key])
            if self.cache:
                self[to_key] = to_data
            return to_data

    def forward_kinematics(self, joints=None, bones=None):
        from_data = {from_key: torch.cat(self[from_key]) for from_key in ["axis_angle", "rot_6D", "root"] if from_key in self}
        packed_sequence = MotionSequence(from_data, "", 0, cache=False)
        packed_out = packed_sequence.forward_kinematics(joints=joints, bones=bones)
        return self.unpack(packed_out)

