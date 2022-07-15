import torch

class CodeDict(torch.nn.Module):
    def __init__(self, key_list, key_type, opts):
        super().__init__()
        self.opts = opts
        self.key_type = key_type
        self.code_dim = self.opts.num_dims
        self.param = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(torch.zeros((1, self.opts.num_dims)))
                for key in key_list
            }
        )

    def __len__(self):
        return len(self.param)

    def get_key_list(self):
        return list(self.param.keys())

    def get_stats(self, motion_batch):
        code_mean_list = [self.param[sample] for sample in motion_batch[self.key_type]]
        code_mean = torch.cat(code_mean_list, 0)
        return code_mean

    def forward(self, motion_batch, **kwargs):
        return {"code": self.get_stats(motion_batch)}

class VariationalCodeDict(torch.nn.Module):
    def __init__(self, key_list, key_type, opts):
        super().__init__()
        self.opts = opts
        self.weight =  self.opts.variational_weight
        self.key_type = key_type
        self.code_dim = self.opts.num_dims

        self.mean_param = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(torch.zeros((1, self.opts.num_dims)))
                for key in key_list
            }
        )

        self.var_param = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(self.opts.logvar_scale * torch.ones((1, self.opts.num_dims)))
                for key in key_list
            }
        )

    def __len__(self):
        return len(self.mean_param)

    def get_stats(self, motion_batch):
        code_mean_list = [self.mean_param[sample] for sample in motion_batch[self.key_type]]
        code_mean = torch.cat(code_mean_list, 0)

        code_logvar_list = [self.var_param[sample] for sample in motion_batch[self.key_type]]
        code_logvar = torch.cat(code_logvar_list, 0)
        return code_mean, code_logvar

    def get_key_list(self):
        return list(self.mean_param.keys())

    def forward(self, motion_batch, **kwargs):
        code_mean, code_logvar = self.get_stats(motion_batch)

        if self.training:
            std = code_logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return {
                "code": eps.mul(std).add_(code_mean),
                "kld": self.kld(code_mean, code_logvar)
            }
        else:
            return {"code": code_mean, "kld": torch.zeros(len(code_mean))}

    def kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) * self.weight

    def avg_mean_distance(self):
        # average distance between code vectors (measure for diversity among codes)
        codes = torch.cat(list(self.mean_param.values()))
        return (codes[:, None, :] - codes[None, :, :]).norm(2, -1).mean().item()