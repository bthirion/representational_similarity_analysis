""" Concurrent analysis of the sensitivity of the statistical tests 
for RSA and linear models respectively.

Author: Bertrand Thirion, Fabian Pedregosa, 2015
"""
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from scipy import linalg
from sklearn import linear_model
from utils import load_data, perm_p_values, square_pdist

# Various parameters
n_samples, n_voxels, n_features = 24, 200, 30
n_repeats = 100
n_perm = 100
model = 'monotonous'
# determines the design matrix. See the documentation of load_data
random_effects = True
# Set random_effects to False/True to trigger a different behavior
# of the simulation.
#
# the semantics of random effects is that the simulated effects either have
# non-zero mean (fixed effects), or zero mean (random effects model)
#
# True: Both models perform equally well
# False: Linear models outperform RSA

pvals_rsa = []
a_space = np.linspace(0, .3, 11)
for a_i in a_space:
    T_ai = []
    # average across repetitions
    for i in range(n_repeats):
        stim, voxels = load_data(
            n_samples, n_voxels, activation=a_i, model=model, seed=i,
            random_effects=random_effects)
        # compute similarity
        stim_similarity = (stim - stim.T) ** 2
        voxels_similarity = - square_pdist(voxels)  # np.corrcoef(voxels)

        # extract lower triangular part of symmetric similarity
        lw_idx = np.triu_indices(n_samples, k=1)
        stim_vsim = stim_similarity[lw_idx]
        voxels_vsim = voxels_similarity[lw_idx]

        # compute the statistic
        T_ai.append(spearmanr(stim_vsim, voxels_vsim)[0])

        T_perm = []
        for _ in range(n_perm):
            # permute the labels
            perm = np.random.permutation(n_samples)
            voxels_vsim_perm = - square_pdist(voxels[perm])[lw_idx]

            # compute the test statistic
            T_perm.append(spearmanr(voxels_vsim_perm, stim_vsim)[0])

    pvals_rsa.append(perm_p_values(T_ai, np.array(T_perm)))


pvals_ls = []
for a_i in a_space:
    T_a_i = []
    T_perm = []

    for i in range(n_repeats):
        stim, voxels = load_data(
            n_samples, n_voxels, activation=a_i, model=model, seed=i,
            random_effects=random_effects)

        clf = linear_model.RidgeCV()
        pred = clf.fit(stim, voxels).predict(stim)
        T_a_i.append(linalg.norm(pred - voxels, 'fro'))
        for _ in range(n_perm):
            # permute the labels
            perm = np.random.permutation(n_samples)
            stim_perm = stim[perm]
            pred_perm = clf.fit(stim_perm, voxels).predict(stim_perm)
            # compute the test statistic
            T_perm.append(linalg.norm(pred_perm - voxels, 'fro'))

    pvals_ls.append(perm_p_values(T_a_i, np.array(T_perm)))

# plot the resulting p-value distributions
from mpltools import special
pvals_rsa = np.array(pvals_rsa)
pvals_ls = np.array(pvals_ls)
power_rsa = 1. * (pvals_rsa < .05)
power_ls = 1. * (pvals_ls < .05)

plt.figure(figsize=(4, 4))
special.errorfill(a_space, np.mean(power_rsa, axis=1),
                  np.std(power_rsa, axis=1), label='RSA')
special.errorfill(a_space, np.mean(power_ls, axis=1),
                  np.std(power_ls, axis=1), label='LS')

plt.xlim((0, np.max(a_space)))
plt.ylim((0, 1.01))
plt.legend(loc=4)
plt.plot(a_space, power_rsa.mean(1), color='b', linewidth=2)
plt.plot(a_space, power_ls.mean(1), color='g', linewidth=2)
plt.ylabel('Power')
plt.xlabel('Effect size')
plt.subplots_adjust(bottom=.15, left=.15)
plt.savefig('power_%s.png' % random_effects)
plt.show()
