import torch

import ImplicitMotion.data.motion_sequences as motions

PI = 3.1415927410125732

def unpack_sequence(seq_lengths, packed_sequence):
    idx_start = 0
    emb_list = []
    for seq_len in seq_lengths:
        emb_list.append(packed_sequence[idx_start:idx_start+seq_len, :])
        idx_start += seq_len
    return emb_list

class MLP(torch.nn.Module):
    def __init__(self, layers, activation, *args, bias, batch_norm=False, **kwargs):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential()

        for i in range(len(layers)-1):
            ni = layers[i]
            no = layers[i+1]
            if i != len(layers)-2:
                self.mlp.add_module("fc{}".format(i), torch.nn.Linear(ni, no, bias=bias))
                if batch_norm:
                    self.mlp.add_module("norm{}".format(i), torch.nn.BatchNorm1d(no))
                self.mlp.add_module("act{}".format(i), activation())
            else:
                self.mlp.add_module("fc{}".format(i), torch.nn.Linear(ni, no, bias=True))

    def forward(self, data):
        return self.mlp(data)

class ImplicitMotionModel(torch.nn.Module):
    def __init__(self, config):
        super(ImplicitMotionModel, self).__init__()
        self.motion_representation = config.model_opts.motion_representation
        self.num_joints = config.dataset_opts.num_joints
        if self.motion_representation == "rot_6D":
            data_dim = self.num_joints * 6 
        elif self.motion_representation == "axis_angle":
            data_dim = self.num_joints * 3
        data_dim += 3 #root joint

        action_ndims = config.action_code_opts.num_dims
        sequence_ndims = config.sequence_code_opts.num_dims
        pos_ndims = config.positional_embedding_opts.num_dims

        if config.action_code_additive:
            assert action_ndims == sequence_ndims,\
                f"Action code,sequence code must have same size. {action_ndims} != {sequence_ndims}"
            code_ndims = action_ndims + pos_ndims
        else:
            code_ndims = action_ndims + sequence_ndims + pos_ndims
        self.latent_dim = code_ndims 
        output_dim = [data_dim - 3 * config.model_opts.root_model]

        self.main = MLP(
            [self.latent_dim] + config.model_opts.layers + output_dim,
            torch.nn.ELU,
            bias=config.model_opts.bias,
            batch_norm=config.model_opts.batch_norm)

        if config.model_opts.root_model:
            self.model_root_layers = [self.latent_dim] + config.model_opts.root_model_layers + [3]
            self.root = MLP(self.model_root_layers, torch.nn.ELU, bias=config.model_opts.bias)
        else:
            self.root = None

    def forward(self, pos_embedding, code, **kwargs):
        # pack sequences
        sequence_lengths = [len(d) for d in pos_embedding]

        code_repeated = torch.cat(
            [code[j, :].expand((seq_len, -1))
            for j, seq_len in enumerate(sequence_lengths)])

        time_concat = torch.cat(pos_embedding)

        predicted_sequence = self.main(
            torch.cat([code_repeated, time_concat], dim=1))

        if self.root is not None:
            predicted_root = self.root(torch.cat([code_repeated, time_concat], dim=1))
            predicted_sequence = torch.cat([predicted_root, predicted_sequence], 1)

        # unpack_sequences
        prediction = unpack_sequence(sequence_lengths, predicted_sequence)

        predicted_sequences = []
        for i, sequence in enumerate(prediction):
            if self.motion_representation == "axis_angle":
                joint_rotations = torch.tanh(sequence[:, 3:]).reshape(-1, self.num_joints, 3) * PI
            elif self.motion_representation == "rot_6D":
                joint_rotations = sequence[:, 3:].reshape(-1, self.num_joints, 6)
            else:
                raise NotImplementedError()

            predicted_sequences.append(
                motions.MotionSequence(
                    {self.motion_representation: joint_rotations, "root": sequence[:, None, :3]},
                    "generated",i))

        return motions.BatchMotionSequence(predicted_sequences)

class TransformerDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.motion_representation = config.model_opts.motion_representation
        self.num_joints = config.dataset_opts.num_joints
        if self.motion_representation == "rot_6D":
            data_dim = self.num_joints * 6 
        elif self.motion_representation == "axis_angle":
            data_dim = self.num_joints * 3
        data_dim += 3 # root joint
        self.data_dim = data_dim

        action_ndims = config.action_code_opts.num_dims
        sequence_ndims = config.sequence_code_opts.num_dims
        pos_ndims = config.positional_embedding_opts.num_dims

        if config.action_code_additive:
            assert action_ndims == sequence_ndims,\
                f"Action code,sequence code must have same size. {action_ndims} != {sequence_ndims}"
            code_ndims = action_ndims
        else:
            code_ndims = action_ndims + sequence_ndims
        
        assert code_ndims == pos_ndims,\
                f"Code and positional embedding must have same size. {code_ndims} != {pos_ndims}"

        self.latent_dim = code_ndims 

        self.activation = "gelu"
        seqTransDecoderLayer = torch.nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=config.model_opts.num_heads,
            dim_feedforward=config.model_opts.ff_size,
            dropout=config.model_opts.dropout,
            activation=self.activation)
        self.seqTransDecoder = torch.nn.TransformerDecoder(seqTransDecoderLayer, num_layers=config.model_opts.num_layers)
        
        self.finallayer = torch.nn.Linear(self.latent_dim, self.data_dim)

    def forward(self, pos_embedding, code, **kwargs):
        try:
            embed = [torch.stack(pos_embedding, 1)]
            code = [code]
        except RuntimeError:
            embed = [embed.unsqueeze(1) for embed in pos_embedding]
            code = code.unsqueeze(1)

        prediction = []
        for iter_embed, iter_code in zip(embed, code):
            output = self.seqTransDecoder(tgt=iter_embed, memory=iter_code.unsqueeze(0))
            
            # output (T time, B batch_size, num_joints * data_dim)
            output = self.finallayer(output)

            # unpack_sequences
            prediction.append(output.permute(1,0,2))

        if len(prediction) == 1:
            prediction = prediction[0]
        else:
            prediction = [pred.squeeze() for pred in prediction]

        predicted_sequences = []
        for i, sequence in enumerate(prediction):
            if self.motion_representation == "axis_angle":
                joint_rotations = torch.tanh(sequence[:, 3:]).reshape(-1, self.num_joints, 3) * PI
            elif self.motion_representation == "rot_6D":
                joint_rotations = sequence[:, 3:].reshape(-1, self.num_joints, 6)
            else:
                raise NotImplementedError()

            predicted_sequences.append(
                motions.MotionSequence(
                    {self.motion_representation: joint_rotations, "root": sequence[:, None, :3]},
                    "generated",i))

        return motions.BatchMotionSequence(predicted_sequences)