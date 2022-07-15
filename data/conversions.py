from functools import reduce

import torch
import pytorch3d.transforms as T

_default_bones = None
def unpack_sequence(seq_lengths, packed_sequence):
    idx_start = 0
    emb_list = []
    for seq_len in seq_lengths:
        emb_list.append(packed_sequence[idx_start:idx_start+seq_len, :])
        idx_start += seq_len
    return torch.stack(emb_list)

def axis_angle_to_joints(bone_hierarchy, joint_angles, root_trajectory=None, joint_locs=None, default_bones=None):
    """
    bone_hierarchy list of lists [from: [to, ...]]
    joint_angles (T, 24, 3): Root orientation + Joint Angles
    root_trajectory (T, 1, 3): Root trajectory
    joint_locs Joint locations to infer bone length
    default_bones: Bone lengths
    """
    if joint_locs is None and default_bones is None:
        raise ValueError("joint_locs and default_bones are None. Bone length unknown.")

    from_idx = reduce(lambda x,y: x+y, [[i] * len(j) for i, j in enumerate(bone_hierarchy) if len(j)!=0])
    to_idx = reduce(lambda x,y: x+y, [j for j in bone_hierarchy if len(j)!=0])

    empty_pose = torch.zeros_like(joint_angles, device=joint_angles.device, dtype=torch.float32)
    global_rot = torch.eye(3, dtype=joint_angles.dtype, device=joint_angles.device)[None, None, :].expand(*joint_angles.shape[:-1], -1, -1).clone()

    rot_mat = T.so3_exponential_map(joint_angles.reshape(-1, 3)).reshape(-1, joint_angles.shape[1], 3, 3)
    rot_mat = rot_mat[:, 0, None, ...] @ rot_mat[:, 1:, ...]

    if joint_locs is None:
        global _default_bones
        if _default_bones is None:
            bones = torch.tensor(default_bones, dtype=torch.float32, device=joint_angles.device)
            _default_bones = bones
        else:
            bones = _default_bones
        bones = bones[None, :, None].repeat(len(joint_angles),1, 1)
    else:
        bones = torch.norm(joint_locs[:, from_idx, :] - joint_locs[:, to_idx, :], 2, -1)[...,None]
    # bones = bones * torch.from_numpy(K.HUMAN_OFFSETS[None, 1:]).to(bones.device)
    bones = bones * torch.tensor([[[0.0, -1.0, 0.0]]], device=joint_angles.device, dtype=torch.float32)
    for i, (x, y) in enumerate(zip(from_idx, to_idx)):
        rot = global_rot[..., x, :, :] @ rot_mat[..., y-1, :, :]
        empty_pose[..., y, :] = empty_pose[..., x, :] + torch.squeeze(rot @ bones[..., i, :, None])
        global_rot[..., y, :, :] = rot

    if root_trajectory is not None:
        empty_pose = empty_pose + root_trajectory
    return empty_pose

def axis_angle_to_matrix(axis_angle):
    return T.axis_angle_to_matrix(axis_angle)

def matrix_to_quaternion(matrix):
    return T.matrix_to_quaternion(matrix)

def quaternion_to_axis_angle(quaternions):
    """
    Check PYTORCH3D_LICENCE before use

    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def batch_smpl_prediction(smpl_model, predictions, use_6D_rot = False):
    pred_lengths = [len(p) for p in predictions]

    pred = torch.cat(predictions)
    if use_6D_rot:
        pred_rots = pred[:, 3:]
        pred_rots = pred_rots.reshape(len(pred_rots), -1, 6)
        pred_rots = quaternion_to_axis_angle(T.matrix_to_quaternion(T.rotation_6d_to_matrix(pred_rots)))
        pred_global_orient = pred_rots[:, 0, :]
        pred_body_pose = pred_rots[:, 1:, :].reshape(len(pred_rots), -1)
    else:
        pred_body_pose = pred[:, 6:]
        pred_global_orient = pred[:, 3:6]
    pred_transl = pred[:, :3]

    smpl_outputs =  smpl_model(body_pose=pred_body_pose, global_orient=pred_global_orient, transl=pred_transl)
    vert_list = unpack_sequence(pred_lengths, smpl_outputs.vertices)
    joint_list = unpack_sequence(pred_lengths, smpl_outputs.joints)
    return {"vertices": vert_list, "joints": joint_list}


def rotation_6d_to_axis_angle(rot):
    return quaternion_to_axis_angle(T.matrix_to_quaternion(T.rotation_6d_to_matrix(rot)))

def axis_angle_to_rotation_6d(rot):
    return T.matrix_to_rotation_6d(T.axis_angle_to_matrix(rot))
