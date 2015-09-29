""" 
Utilities for the RSA experiment: stim generation etc.
"""
import numpy as np
from sklearn import preprocessing
from scipy import stats


def square_pdist (x):
    """ redturn distance matrix"""
    # return np.sqrt(((x[:, np.newaxis] - x[np.newaxis]) ** 2).sum(2))
    return 1 - np.corrcoef(x)


def load_data(n_samples, n_voxels, activation=0, n_features=1,
              heteroscedastic=False, shuffle=False, model='noisy', seed=1,
              random_effects=True):
    """ generate some data for the exeriments"""
    np.random.seed([seed])
    # Stim generation: three possibilities
    if model == 'monotonous':
        stim = np.arange(n_samples).astype(np.float)[:, None]
    elif model == 'noisy':
        stim = np.random.randn(n_samples, n_features) + np.repeat(
            np.arange(n_samples)[:, np.newaxis], n_features, 1).astype(
            np.float)
    elif model == 'binary':
        stim = np.hstack((np.zeros(12), np.ones(12)))[:, np.newaxis]
    else:
        raise ValueError('Unknown model')
    if shuffle:
        np.random.shuffle(stim)
    stim = preprocessing.StandardScaler().fit_transform(stim)

    # simulate the noise
    noise = np.random.randn(n_samples, n_voxels)
    if heteroscedastic:
        noise[n_samples / 2:] *= np.sqrt(2)
        # noise[:n_voxels / 2] *= 2

    if random_effects:
        effects = activation * np.random.randn(1, n_voxels)
    else:
        effects = activation * np.random.rand(1, n_voxels)

    voxels = effects * stim + noise
    # normalize both stim and voxels
    voxels = preprocessing.StandardScaler().fit_transform(voxels)
    return stim, voxels


def perm_p_values(samples, distribution):
    pvals_ = np.zeros_like(samples)
    for i in range(len(samples)):
        pvals_[i] = stats.percentileofscore(distribution, samples[i])
    return pvals_ / 100
