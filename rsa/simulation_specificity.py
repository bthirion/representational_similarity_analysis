""" Simulation-based analysis of the specificity of statistical tests

The p-values of statistical tests of association should be distributed
uniformly over the [0, 1] interval whenever no effect is simulated,
which is the case in this script.

However, when the noise covariance matrix has some latent structure,
for some tests the control can be biased, resulting into a uniform p-values
distribution.

Author: Bertrand Thirion, Fabian Pedregosa, 2015.
"""
import matplotlib.pyplot as plt

import numpy as np
from sklearn import preprocessing, linear_model

from scipy.stats import spearmanr, percentileofscore
from sklearn import preprocessing, linear_model
from utils import load_data, square_pdist

do_save = False
n_iter = 100
# perform a permutation test and return the p-value.
# This will be repeated n_iter times to find the distributed of p-values.
# the seed is given as argument to make sure that
# the data for the different runs is indeed different
# (something that is not obvious otherwise because of multiprocessing)

n_samples, n_features, n_draws = 24, 200, 1000
model = 'monotonous'
heteroscedastic=True


def compute_pval_rsa(seed):
    stim, voxels = load_data(n_samples, n_features, model=model, seed=seed,
                             heteroscedastic=heteroscedastic)

    # compute similarity
    stim_ = stim
    if stim.shape[1] == 1:
        stim_ = np.hstack((stim, - stim))

    stim_similarity = square_pdist(stim_)  # np.corrcoef(stim_)
    voxels_similarity = square_pdist(voxels)  # np.corrcoef(voxels)

    # indices to extract lower triangular part of a matrix
    lw_idx = np.triu_indices(n_samples, k=1)

    stim_vsim = stim_similarity[lw_idx]
    voxels_vsim = voxels_similarity[lw_idx]

    # compute the statistic
    # T = np.corrcoef(stim_vsim, voxels_vsim)[0, 1]
    T = spearmanr(voxels_vsim, stim_vsim)[0]
    T_perm = []
    for i in range(n_draws):
        # permute the labels
        perm = np.random.permutation(n_samples)
        # voxels_vsim_perm = np.corrcoef(voxels[perm])[lw_idx]
        voxels_vsim_perm = square_pdist(voxels[perm])[lw_idx]
        # compute the test statistic
        # T_perm.append(np.corrcoef(voxels_vsim_perm, stim_vsim)[0, 1])
        T_perm.append(spearmanr(voxels_vsim_perm, stim_vsim)[0])

    pval = 1 - percentileofscore(np.array(T_perm), T) / 100.
    return pval


def compute_pval_ls(seed):
    np.random.seed(seed)
    stim, voxels = load_data(n_samples, n_features, model=model, seed=seed,
                             heteroscedastic=heteroscedastic)

    clf = linear_model.LinearRegression()
    pred = clf.fit(stim, voxels).predict(stim)
    T = np.linalg.norm(pred - voxels, 'fro')
    T_perm = []
    for _ in range(n_draws):
        # permute the labels
        perm = np.random.permutation(n_samples)
        stim_perm = stim[perm]
        pred_perm = clf.fit(stim_perm, voxels).predict(stim_perm)
        # compute the test statistic
        T_i = np.linalg.norm(pred_perm - voxels, 'fro')
        T_perm.append(T_i)
    pval = percentileofscore(T_perm, T) / 100.
    return pval


# compute permuted p-values in parallel
from joblib import Parallel, delayed
all_pval_rsa = Parallel(n_jobs=2)(delayed(compute_pval_rsa)(seed)
                                  for seed in range(n_iter))
np.save('pval_rsa.npy', all_pval_rsa)


# compute permuted p-values in parallel
all_pval_ls = Parallel(n_jobs=2)(delayed(compute_pval_ls)(seed)
                                 for seed in range(n_iter))
np.save('pval_ls.npy', all_pval_ls)

bins = 20
ref = n_iter / bins
plt.figure(figsize=(6, 2))
plt.subplot(121)
all_pval_rsa = np.load('pval_rsa.npy')
plt.title('Distribution of p-values for RSA')
plt.hist(all_pval_rsa, bins=bins, color='g')
plt.plot([0, 1], [ref, ref], 'r', linewidth=2)
plt.subplot(122)
all_pval_ls = np.load('pval_ls.npy')
plt.title('Distribution of p-values for LS')
plt.hist(all_pval_ls, bins=bins, color='g')
plt.plot([0, 1], [ref, ref], 'r', linewidth=2)
if do_save:
    plt.savefig('rsa_specificity_hetero.png')

plt.show()
