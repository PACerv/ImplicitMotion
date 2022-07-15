import math

import torch

class FixedPositionalEmbeddingFunction(torch.nn.Module):
    def __init__(self, key_list, key_type, opts, max_len = 4000):
        # taken from "Attention is all you need" and https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        super(FixedPositionalEmbeddingFunction, self).__init__()
        self.num_dims = opts.num_dims
        pe = torch.zeros(max_len, self.num_dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.num_dims, 2).float() * (-math.log(opts.num_freq) / self.num_dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        

    def forward(self, sequence_options, **kwargs):
        return {
            "time_sequences": self.predict(None, sequence_options["time_idx"])
        }

    def predict(self, code, time_steps):
        return [self.pe[steps.long(), :].squeeze() for steps in time_steps]
