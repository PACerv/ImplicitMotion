"""
Adopted from https://github.com/Mathux/ACTOR/tree/master/src/recognition/models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pytorch3d.transforms as T

from ImplicitMotion.test.stgcnutils.tgcn import ConvTemporalGraphical
from ImplicitMotion.test.stgcnutils.graph import Graph
__all__ = ["STGCN"]


class STGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, device, **kwargs):
        super().__init__()

        self.device = device
        self.num_class = num_class
        
        self.losses = ["accuracy", "cross_entropy", "mixed"]
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, batch):
        # TODO: use mask
        # Received batch["x"] as
        #   Batch(48), Joints(23), Quat(4), Time(157
        # Expecting:
        #   Batch, Quat:4, Time, Joints, 1
        motion_subsequences = [seq.get_subsequence(60) for seq in batch]
        motion_sequence_joints = torch.stack([seq["rot_6D"].reshape(-1, 144) for seq in motion_subsequences], 0)

        x = motion_sequence_joints.reshape(-1, 60, 24, 6).permute(0, 3, 1, 2).unsqueeze(4).contiguous()
        # x = batch["x"].permute(0, 2, 3, 1).unsqueeze(4).contiguous()

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # compute feature
        # _, c, t, v = x.size()
        # features = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        # batch["features"] = features
        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # features
        feat = x.squeeze()
        
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x, feat

    def compute_accuracy(self, batch):
        confusion = torch.zeros(self.num_class, self.num_class, dtype=int)
        yhat = batch["yhat"].max(dim=1).indices
        ygt = batch["y"]
        for label, pred in zip(ygt, yhat):
            confusion[label][pred] += 1
        accuracy = torch.trace(confusion)/torch.sum(confusion)
        return accuracy
    
    def compute_loss(self, batch):
        cross_entropy = self.criterion(batch["yhat"], batch["y"])
        mixed_loss = cross_entropy
        
        acc = self.compute_accuracy(batch)
        losses = {"cross_entropy": cross_entropy.item(),
                  "mixed": mixed_loss.item(),
                  "accuracy": acc.item()}
        return mixed_loss, losses


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, motion_length, use_fixed_length):
        self.motion_length = motion_length
        self.use_fixed_length = use_fixed_length
        remove_short_sequences = lambda x: len(x.poses) > (0.75 * motion_length)
        remove_short_sequences_smpl = lambda x: len(x.motion_data) > (0.75 * motion_length)
        try:
            self.dataset = list(filter(remove_short_sequences, dataset))
        except AttributeError:
            self.dataset = list(filter(remove_short_sequences_smpl, dataset))


    def __getitem__(self, item):
        data = self.dataset[item]
        try:
            motion = T.matrix_to_rotation_6d(T.axis_angle_to_matrix(data.poses.reshape(-1, 24, 3)))
        except AttributeError:
            motion = data.motion_data.reshape(-1, 24, 6)
        label = data.label
        motion_len = motion.shape[0]
        # Motion can be of various length, we randomly sample sub-sequence
        # or repeat the last pose for padding

        # random sample
        if self.use_fixed_length:
            if motion_len >= self.motion_length:
                gap = motion_len - self.motion_length
                start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
                end = start + self.motion_length
                r_motion = motion[start:end]
                # offset deduction
                # r_motion = r_motion - r_motion[0, 0, None, None, :]
            # padding
            else:
                gap = self.motion_length - motion_len
                pad_poses = motion[-1].unsqueeze(0).repeat(gap, 1, 1)
                r_motion = torch.cat([motion, pad_poses], 0)
                # last_pose = np.expand_dims(motion[-1], axis=0)
                # pad_poses = np.repeat(last_pose, gap, axis=0)
                # r_motion = np.concatenate([motion, pad_poses], axis=0)
            motion = r_motion
        return motion, label

    def __len__(self):
        return len(self.dataset)