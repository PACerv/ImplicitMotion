import logging

import numpy as np

from sklearn.mixture import GaussianMixture
import torch

EPS = 1e-6


class LengthConditionalDistribution(object):
    def __init__(self,
        min_window=10, min_pop=50, overlap=0, cov_type="full", adjust_min_pop=False,
        threshold=30, saturation_point=50, saturation_minimum=10, gaussian_mixture=0):
        super().__init__()
        self.min_window = min_window
        self.min_pop = min_pop
        self.overlap = overlap
        self.cov_type = cov_type
        self.gaussian_mixture = gaussian_mixture
        # only relevant if adjust_min_pop True
        self.adjust_min_pop = adjust_min_pop
        self.threshold = threshold
        self.saturation_point = saturation_point
        self.saturation_minimum = saturation_minimum

    def adjust(self, window):
        if window <= self.threshold:
            min_pop = self.min_pop
        elif self.threshold < window <= self.saturation_point:
            min_pop = ((window - self.threshold) * (self.saturation_minimum - self.min_pop)/(self.saturation_point - self.threshold)) + self.min_pop
        else:
            min_pop = self.saturation_minimum
        return min_pop

    def fit(self, codes, length_list):
        self.dims = codes.shape[-1]
        length_list = torch.tensor(length_list)
        self.min_len = length_list.min().item()
        self.max_len = length_list.max().item()
        # Compute histogram
        histogram = torch.zeros(length_list.max())
        for length in length_list:
            histogram[length-1] += 1

        # Fit intervals
        interval_list = []
        pop_list = []
        

        while len(interval_list) == 0 or interval_list[-1][1] < length_list.max():
            # init new interval
            if len(interval_list) == 0:
                interval_start = self.min_len
            else:
                interval_start = interval_list[-1][1] - self.overlap

            min_pop = self.min_pop
            pop = 0
            window = 0
            while (pop < min_pop or window < self.min_window) and (interval_start + window) < length_list.max():
                pop += histogram[interval_start + window].item()
                window += 1
                if self.adjust_min_pop:
                    min_pop = self.adjust(window)

            interval_end = interval_start + window

            interval_list.append((interval_start, interval_end))
            pop_list.append(pop)

        # last interval may not have min_pop. Increase overlap to match
        
        if len(interval_list) > 1:
            window = interval_list[-1][1] - interval_list[-1][0]
            if self.adjust_min_pop:
                min_pop = self.adjust(window)
            else:
                min_pop = self.min_pop
            new_last_interval_start = interval_list[-1][0]
            
            while pop_list[-1] < min_pop:
                new_last_interval_start -= 1
                window += 1
                if self.adjust_min_pop:
                    min_pop = self.adjust(window)
                pop_list[-1] += histogram[new_last_interval_start].item()

            interval_list[-1] = (new_last_interval_start, interval_list[-1][1])

        distributions = []
        distribution_stats = []

        for interval in interval_list:
            idx = ((interval[0]+1) <= length_list) * (length_list < (interval[1]+1))
            interval_codes = codes[idx, :]

            n_components = min(self.gaussian_mixture, len(interval_codes))
            num_overfitted = 1
            while num_overfitted != 0 and n_components > 0:
                seed = 0
                max_attempts = 10
                num_overfitted = 1
                fitting_codes = interval_codes.cpu().detach().numpy()
                overfitting_threshold = np.linalg.norm(fitting_codes,2,-1).mean() * 0.1
                while num_overfitted != 0 and seed < max_attempts:
                    seed += 1
                    gmm = GaussianMixture(n_components=n_components, covariance_type=self.cov_type, random_state=seed)
                    gmm.fit(fitting_codes)

                    # Overfitting check
                    num_overfitted = (np.linalg.norm(fitting_codes[None, ...] - gmm.means_[:, None, ...],2 ,-1) < overfitting_threshold).sum()

                if num_overfitted != 0:
                    n_components -= 1

            if n_components <= 1:
                gmm.weights_[0] = 1.
            distributions.append(gmm)
            distribution_stats.append({
                "num_overfitting": num_overfitted,
                "converged": gmm.converged_,
                "gmm_score": gmm.score(fitting_codes),
                "num_components": n_components,
                "seed": seed
            })
            if not gmm.converged_:
                logging.warn("GMM at interval %s didn't converge", str(interval))

        self.pop_list = pop_list
        self.interval_list = interval_list
        self.interval_median = torch.tensor([(interval[0] + interval[1])/2 for interval in self.interval_list])
        self.distributions = distributions
        self.distribution_stats = distribution_stats

    def sample_n(self, lengths, device="cpu"):
        dist_idx = (lengths.unsqueeze(1) - self.interval_median.unsqueeze(0)).abs().min(1)[1]
        
        new_codes = torch.zeros(len(lengths), self.dims, device=device)
        for i, dist in enumerate(self.distributions):
            part_idx = dist_idx == i
            if part_idx.sum() == 0: continue
            if isinstance(dist, GaussianMixture):
                sampled_codes = torch.from_numpy(dist.sample(part_idx.sum())[0]).float().to(device)
            else:
                sampled_codes = dist.sample((part_idx.sum(),))

            new_codes[part_idx, :] = sampled_codes

        return new_codes

    def __str__(self):
        return "LengthConditionalDistribution: \n" +\
            "\n".join([f"{inter}: pop {int(pop):d}  overfitting {stats['num_overfitting']}  score {stats['gmm_score']}  seed: {stats['seed']}  components: {stats['num_components']}" \
                for pop, inter, stats in zip(self.pop_list, self.interval_list, self.distribution_stats)])

class GaussianMixtureTorch(object):
    def __init__(self, n_components, covariance_type, len_list=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        
        if len_list is not None:
            self.len_list = torch.tensor(len_list)
            self.min_len = self.len_list.min().item()
            self.max_len = self.len_list.max().item()

    def fit(self, code):
        if hasattr(code, "device"):
            code = code.cpu().detach().numpy()

        num_overfitted = 1
        overfitting_threshold = np.linalg.norm(code,2,-1).mean() * 0.1
        max_attempts = 50
        seed = 0
        while num_overfitted != 0 and seed < max_attempts:
            seed += 1
            gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
            gmm.fit(code)

            # Overfitting check
            num_overfitted = (np.linalg.norm(code[None, ...] - gmm.means_[:, None, ...],2 ,-1) < overfitting_threshold).sum()

        self.gmm = gmm
        if self.n_components == 1:
            self.gmm.weights_[0] = 1.
        self.stats = {
            "population": len(code),
            "num_overfitting": num_overfitted,
            "converged": self.gmm.converged_,
            "gmm_score": self.gmm.score(code),
            "seed": seed
        }

    def __str__(self):
        return f"GaussianMixtureModel --- population: {self.stats['population']}  overfitting: {self.stats['num_overfitting']}   score: {self.stats['gmm_score']} seed: {self.stats['seed']}"

    def sample_n(self, n, device="cpu"):
        code = torch.from_numpy(self.gmm.sample(n)[0]).float().to(device)
        return code

class RandomSampler(object):
    def __init__(self):
        super().__init__()

    def fit(self, codes):
        self.codes = codes

    def sample_n(self, num_samples, device="cpu"):
        idx = torch.ones(len(self.codes)).multinomial(num_samples, replacement=True)
        codes = self.codes[idx].to(device)
        return codes

class VariableLengthRandomSampler(object):
    def __init__(self):
        super().__init__()

    def fit(self, codes, lengths):
        self.codes = codes.detach().clone()
        self.length_list = torch.tensor(lengths)
        self.min_len = self.length_list.min().item()
        self.max_len = self.length_list.max().item()

    def sample_n(self, lengths, device="cpu"):
        dist_idx = (lengths.unsqueeze(0) - self.length_list.unsqueeze(1)).abs().min(0)[1]
        
        codes = torch.stack([self.codes[idx, ...].detach().clone().contiguous().to(device) for idx in dist_idx])
        return codes, self.length_list[dist_idx]
