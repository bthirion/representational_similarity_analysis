""" This script analyses the results of the Haxby experiment.

(Figure generation etc.)

Author: Bertrand Thirion, 2015.
"""

import numpy as np
import matplotlib.pyplot as plt
from nibabel import load
from nilearn import datasets, image, input_data, plotting
from scipy.stats import spearmanr

rsa = np.maximum(
    0, - np.log10(1. + 1.01e-4 - np.load('roi_scores_rsa_0.npz')['arr_0']))
lm = np.maximum(
    0, - np.log10(1. + 1.01e-4 - np.load('roi_scores_lm_0.npz')['arr_0']))
rrr = np.maximum(
    0, - np.log10(1. + 1.01e-4 - np.load('roi_scores_rrr_0.npz')['arr_0']))
classif = np.load('roi_scores_classif_0.npz')['arr_0']


print('rsa', spearmanr(classif, rsa)[0])
print('lm', spearmanr(lm, classif)[0])
print('rrr', spearmanr(rrr, classif)[0])


# Scatter plot of the scores ###################################

plt.figure()
plt.plot(rsa, lm, 'o', color='b')
plt.plot(rsa, rrr, 'o', color='g')
plt.plot(rsa, rsa, 'k')
plt.xlabel('RSA')
plt.ylabel('Linear model')
plt.title("Log p-value")
plt.savefig('scatter_plot.png')


# Create Images ################################################

atlas_filename, labels = datasets.fetch_harvard_oxford(
    'cort-maxprob-thr25-2mm', symmetric_split=True)

haxby_dataset = datasets.fetch_haxby_simple()
func_filename = haxby_dataset.func
mask_filename = haxby_dataset.mask
#mean = image.mean_img(func_filename)

affine = load(mask_filename).get_affine()
shape = load(mask_filename).get_shape()
atlas = image.resample_img(atlas_filename, target_affine=affine,
                           target_shape=shape, interpolation='nearest')
roi_masker = input_data.NiftiLabelsMasker(labels_img=atlas,
                                          mask_img=mask_filename)
roi_masker.fit(mask_filename)  # just to have it fitted

cut_coords = (30, -45, -12)

roi_score_img = roi_masker.inverse_transform(rsa[np.newaxis])
plotting.plot_stat_map(roi_score_img, title='RSA', cut_coords=cut_coords)
plt.savefig('rsa.png')
roi_score_img = roi_masker.inverse_transform(lm[np.newaxis])
plotting.plot_stat_map(roi_score_img, title='Linear model',
                       cut_coords=cut_coords)
plt.savefig('lm.png')

plotting.plot_roi(atlas, title="Harvard Oxford atlas", cut_coords=cut_coords)
# print labels

from scipy.stats import fligner
X = roi_masker.transform(func_filename)
y, session = np.loadtxt(haxby_dataset.session_target).astype('int').T
conditions = np.recfromtxt(haxby_dataset.conditions_target)['f0']
non_rest = conditions != b'rest'
conditions = conditions[non_rest]
y, session = y[non_rest], session[non_rest]
y = y[session < 4]

var_stat = np.zeros(X.shape[1])
for j, x in enumerate(X.T):
    _, var_stat[j] = fligner(
        x[y == 8], x[y == 1], x[y == 2], x[y == 3],
        x[y == 4], x[y == 5], x[y == 6], x[y == 7])

var_img = roi_masker.inverse_transform(
    - np.hstack((0, np.log10(var_stat)))[np.newaxis])
plotting.plot_stat_map(var_img, cut_coords=cut_coords, title='Fligner test',
                       vmax=4)
plt.savefig('var_stat.png')


plt.show()
