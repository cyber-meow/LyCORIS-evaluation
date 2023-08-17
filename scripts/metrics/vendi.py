"""
Modified from
https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/metrics/vendit.py
All credits go to the original authors
"""

from sklearn import preprocessing
from sklearn.metrics.pairwise import polynomial_kernel
import scipy
import scipy.linalg
import numpy as np


def compute_vendi_score(X, q=1, normalize=True, kernel='linear'):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    if kernel == 'linear':
        S = X @ X.T
    elif kernel == 'polynomial':
        S = polynomial_kernel(X, degree=3, gamma=None, coef0=1)
    else:
        raise NotImplementedError("kernel not implemented")
    # print('similarity matrix of shape {}'.format(S.shape))
    w = scipy.linalg.eigvalsh(S / n)
    return np.exp(entropy_q(w, q=q))


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_**q).sum()) / (1 - q)


def compute_per_prompt_vendi_scores(reps, n_images_per_prompt):
    vendi_per_prompt = []
    n_prompts = len(reps)
    for start in range(0, n_prompts, n_images_per_prompt):
        end = min(start+n_images_per_prompt, n_prompts)
        reps_prompt = reps[start:end]
        vendi_per_prompt.append(compute_vendi_score(reps_prompt))
    return np.array(vendi_per_prompt)
