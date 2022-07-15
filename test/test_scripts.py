import torch

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

import ImplicitMotion.models.time_function as time
import ImplicitMotion.models.code_dict as CodeDict

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138/5
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    vals = torch.view_as_complex(vals.contiguous())
    vals_pow = vals.pow(p)
    vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs + 1e-6 * torch.eye(vecs.shape[0], device=vecs.device))))
    return matrix_pow

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    # https://stackoverflow.com/questions/64554658/calculate-covariance-of-torch-tensor-2d-feature-map
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def sample_time_function(time_function, device, time_steps = 1000, num_lines = 50):
    if isinstance(time_function, time.HyperSIRENTimeFunction) or isinstance(time_function, time.CosineTimeFunction):
        filelist = list(time_function.time_codes.param.keys())
    else:
        raise NotImplementedError

    ridx = torch.ones(len(filelist)).multinomial(min(num_lines, len(filelist)))

    input_file_names = [filelist[idx] for idx in ridx]

    sequence_options = {
        "filenames": input_file_names,
        "time_steps": [torch.arange(1,time_steps+1, device=device) for _ in input_file_names]
    }

    return time_function(sequence_options), input_file_names

def sample_time_function_experiment(experiments, device, time_steps = 1000, num_lines = 50):
    dummy_time = list(experiments.values())[0]["time_function"]
    if isinstance(dummy_time, time.HyperSinusoidalTimeFunction) or isinstance(dummy_time, time.HyperLinearTimeFunction):
        if isinstance(dummy_time.time_codes, CodeDict.CodeDict):
            filelist = list(dummy_time.time_codes.param.keys())
        elif isinstance(dummy_time.time_codes, CodeDict.VariationalCodeDict):
            filelist = list(dummy_time.time_codes.mean_param.keys())
    else:
        raise NotImplementedError
    
    ridx = torch.ones(len(filelist)).multinomial(min(num_lines, len(filelist)))

    input_file_names = [filelist[idx] for idx in ridx]

    sequence_options = {
        "filenames": input_file_names,
        "time_steps": [torch.arange(1,time_steps+1, device=device) for _ in input_file_names]
    }

    return {key: expr["time_function"](sequence_options)["time_sequences"] for key, expr in experiments.items()}, input_file_names

def pca_transform(data):
    pipeline = Pipeline([('scaling', Normalizer()), ('pca', PCA(n_components=2))])
    pca = pipeline.fit(data)
    pca_codes = pca.transform(data)
    return pca, pca_codes

def lda_transform(data, labels):
    pipeline = Pipeline([('scaling', Normalizer()), ('lda', LDA(n_components=2))])
    pca = pipeline.fit(data, labels)
    pca_codes = pca.transform(data)
    return pca, pca_codes

def gmm_preprocess(n_feats, covariance_type, code, labels):
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    pipeline = Pipeline([('scaling', Normalizer()), ('pca', PCA(n_components=n_feats))])
    pca = pipeline.fit(code)
    pca_codes = pca.transform(code)

    # mean by label
    codes_by_action = {}
    for a in np.unique(labels):
        #sorted by default
        codes_by_action[a.item()] = pca_codes[np.nonzero(labels==a),:]

    mean_init = np.stack([np.squeeze(code).mean(0) for code in codes_by_action.values()],0)

    # covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}
    if covariance_type == "full":
        action_wise_var = np.stack([np.diag(np.squeeze(code).var(0)) for code in codes_by_action.values()],0)
        precision_init = np.linalg.inv(action_wise_var)
    elif covariance_type == "tied":
        action_wise_var = np.stack([np.squeeze(code).var(0) for code in codes_by_action.values()],0)
        precision_init = np.linalg.inv(np.diag(action_wise_var.mean(0)))
    elif covariance_type == "diag":
        action_wise_var = np.stack([np.squeeze(code).var(0) for code in codes_by_action.values()],0)
        precision_init = action_wise_var ** -1
    elif covariance_type == "spherical":
        action_wise_var = np.stack([np.squeeze(code).var(0) for code in codes_by_action.values()],0)
        precision_init = action_wise_var.max(1) ** -1
    return pca, pca_codes, mean_init, precision_init

def code_gmm(code, labels, ndims=[100], cov_types=["full"]):
    acc_mat = np.zeros((len(ndims), len(cov_types)))
    gmms = []
    pcas = []

    for i, n_components in enumerate(ndims):
        gmms.append([])
        pcas.append([])
        for j, covariance_type in enumerate(cov_types):
            pca, pca_codes, mean_init, precision_init = gmm_preprocess(n_components, covariance_type, code.detach().cpu().numpy(), labels.numpy())
            gmm = GaussianMixture(len(labels.unique()), covariance_type=covariance_type, means_init=mean_init, precisions_init=precision_init, max_iter=2000)
            try:
                gmm.fit(pca_codes)
                y_action = gmm.predict(pca_codes)
                gmm_action_accuracy = (y_action == labels.numpy()-1).sum() / len(labels)
                acc_mat[i][j] = gmm_action_accuracy
            except Exception:
                gmm = None
                pca = None
            gmms[i].append(gmm)
            pcas[i].append(pca)

    return pcas, gmms, acc_mat
