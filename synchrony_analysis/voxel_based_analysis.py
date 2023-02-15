from nilearn import plotting
from nilearn.image import resample_to_img
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import (sample_rate_fmri, intermediate_sample_rate, clean_level, main_project_path)
import nibabel as nib
from scipy import signal
from matplotlib import pyplot as plt
import os
import pandas as pd
import argparse
from utils.gastric_utils import to_phase_resampled, plot_example_synchrony
import numpy as np

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='subject_name', help='The subject to process')
parser.add_argument('run', metavar='run_number', help='The run number of the subject')
args = parser.parse_args()
subject_name = args.subject
run = str(args.run)

print('Processing subject:', subject_name, 'run:', run)

#load data
data_path = '../../derivatives/brain_gast/' + subject_name + '/' + subject_name+run
MNI_tamplate_path = os.environ['FSL_DIR'] + '/data/standard/MNI152_T1_2mm.nii.gz'
gastric_signal = np.load(data_path + '/gast_data_' + subject_name + '_run' + run + clean_level +
                         '_sliced.npy')
brain_signal = np.load(data_path + '/func_filtered_' + subject_name + '_run' + run + clean_level +
                       '_sliced.npz')['brain_signal']
original_fmri = nib.load(f'{main_project_path}/fmriprep/out/{subject_name}/fmriprep/{subject_name}_task-rest_run-0{run}'
               f'_space-MNI152NLin6Asym_desc-preproc_bold_{clean_level}.nii.gz')
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
record_meta = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name) &
                                    (record_meta_pd['run'] == int(run)),:].to_dict('records')[0]

MNI_tamplate_3mm = resample_to_img(nib.load(MNI_tamplate_path), original_fmri)
mask = np.load('../../derivatives/brain_gast/mask_' + subject_name + '_run' + run + clean_level + '.npz')['mask']
plot_path =  '../../plots/brain_gast/' + subject_name + '/' + subject_name+run

# calculate the PLV
# see Time series analysis in neuroscience. Alexander Zhigalov Dept. of CS, University of Helsinki and Dept. of NBE,
# Aalto University

# calc phase
brain_signal_phase = signal.hilbert(brain_signal, axis=1)
brain_signal_phase = np.apply_along_axis(np.angle, 1, brain_signal_phase)
gastric_signal_phase = to_phase_resampled(gastric_signal, intermediate_sample_rate, sample_rate_fmri)

# calc phase-locking value
plvs_empirical = np.abs(np.mean(np.exp(1j * (brain_signal_phase - gastric_signal_phase[np.newaxis,:])), axis=1))

vol_new = np.zeros((original_fmri.shape))
vol_new[mask] = plvs_empirical
img_plv = nib.Nifti1Image(vol_new, affine = original_fmri.affine, header=original_fmri.header)

# plot an example of high/ low/ random gastric-brain synchrony
plot_example_synchrony(gastric_signal, brain_signal, plvs_empirical, plot_path + 'egg_BOLD_sync_example.png')

# plot the empirical PLV map
plotting.plot_stat_map(img_plv, bg_img = MNI_tamplate_3mm, title="plot_stat_map",colorbar = True, threshold=np.percentile(plvs_empirical,95))
plt.savefig(plot_path + 'empirical_plv_map.png', dpi=200)
plt.close('all')

print('calculating null distribution of PLV values')
sample_per_min = int(intermediate_sample_rate * 60)
samples_per_tr = int(intermediate_sample_rate / sample_rate_fmri)
k = int((len(gastric_signal) - (2*sample_per_min)) / samples_per_tr)
plvs_permutation = np.zeros((len(plvs_empirical), k))
for inx_permut in np.arange(k):
    # permut the gastric signal
    gastric_signal_permut = np.roll(gastric_signal, sample_per_min + int(inx_permut * samples_per_tr))
    # calc phase
    gastric_phase_permut = to_phase_resampled(gastric_signal_permut,
                                              intermediate_sample_rate, sample_rate_fmri)
    # calc phase-locking value
    plvs_permutation[:,inx_permut] = \
        np.abs(np.mean(np.exp(1j * (brain_signal_phase - gastric_phase_permut[np.newaxis, :])), axis=1))

# Compute all the relevant subject-level statistical maps
plv_p_vals = (plvs_permutation < plvs_empirical[:,np.newaxis]).mean(axis=1)
plv_permut_median = np.median(plvs_permutation,axis=1)
plv_delta = plvs_empirical - plv_permut_median

for measure_name, measure in zip(['plv_p_vals', 'plv_delta', 'plv_permut_median', 'plvs_empirical'], [plv_p_vals, plv_delta, plv_permut_median, plvs_empirical]):
    vol_new = np.zeros((original_fmri.shape))
    vol_new[mask] = measure
    img_plv = nib.Nifti1Image(vol_new, affine = original_fmri.affine, header=original_fmri.header)
    nib.save(img_plv, data_path + '/' + measure_name + '_' + subject_name + '_run' + run + clean_level + '.nii.gz')

    plotting.plot_stat_map(img_plv, bg_img = MNI_tamplate_3mm, title=measure_name,colorbar = True,
                           threshold=np.percentile(plvs_empirical,95))
    plt.savefig(plot_path + 'thres95_' + measure_name + '_map.png', dpi=200)
    plt.close('all')

print('Done synchrony analysis for: ', subject_name, run)
