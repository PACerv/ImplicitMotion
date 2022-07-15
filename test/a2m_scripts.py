"""
Adopted from https://github.com/EricGuo5513/action-to-motion/blob/3cc99f7887fb4839caf8126f293aa9208980136d/models/motion_gan.py
"""

import torch
import numpy as np
import ImplicitMotion.test.test_scripts as test
from scipy import linalg

class MotionDiscriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size=1, use_noise=None):
        super(MotionDiscriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = torch.nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = torch.nn.Linear(hidden_size, 30)
        self.linear2 = torch.nn.Linear(30, output_size)

    def forward(self, motion_sequences, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        motion_subsequences = [seq.get_subsequence(60) for seq in motion_sequences]
        try:
            motion_sequence = torch.stack([seq["joints"].reshape(-1, self.input_size) for seq in motion_subsequences], 1)
        except:
            motion_sequence_joints = torch.stack([seq["joints"].reshape(-1, self.input_size-3) for seq in motion_subsequences], 1)
            motion_sequence_root = torch.stack([seq["root"].reshape(-1, 3) for seq in motion_subsequences], 1)
            motion_sequence = torch.cat([motion_sequence_root, motion_sequence_joints], -1)
        motion_sequence = motion_sequence.reshape(60, -1, int(self.input_size/3), 3)
        motion_sequence = (motion_sequence - motion_sequence[0, None, :, 0, None, :]).reshape(60, -1, self.input_size)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer, motion_sequence.device)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2, lin1

    def initHidden(self, num_samples, layer, device):
        return torch.zeros(layer, num_samples, self.hidden_size, device=device, requires_grad=False)
        # return torch.randn(layer, num_samples, self.hidden_size, device=device, requires_grad=False)

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, motion_length, fixed_length):
        self.dataset = dataset
        self.motion_length = motion_length
        self.fixed_length = fixed_length

    def __getitem__(self, item):
        data = self.dataset[item]
        motion = data.motion_data
        label = data.label
        motion_len = motion.shape[0]
        # Motion can be of various length, we randomly sample sub-sequence
        # or repeat the last pose for padding

        # random sample
        if self.fixed_length:
            if motion_len >= self.motion_length:
                gap = motion_len - self.motion_length
                start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
                end = start + self.motion_length
                r_motion = motion[start:end]
                # offset deduction
                r_motion = r_motion - r_motion[0, 0, None, None, :]
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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = test._matrix_pow(sigma1 @ sigma2, 0.5)
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = test._matrix_pow((sigma1+offset) @ (sigma2+offset), 0.5)
        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        if not np.allclose(torch.diagonal(covmean).imag, 0, atol=1e-3):
            m = torch.abs(covmean.imag).max()
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return ((diff @ diff ) + torch.trace(sigma1) +
            torch.trace(sigma2) - 2 * tr_covmean)

def calculate_frechet_distance_numpy(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_diversity_multimodality(activations, labels, num_labels):
    #### print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    #### print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity, multimodality

