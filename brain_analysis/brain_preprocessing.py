import nibabel as nib
from matplotlib import pyplot as plt
from nilearn import image, plotting
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import (sample_rate_fmri, bandpass_lim, filter_order, transition_width,
                    brain_fwhm, main_project_path, zscore_flag, clean_level, clean_level)
import numpy as np
from mne.filter import filter_data
from sklearn.preprocessing import scale
import pandas as pd
import argparse
import os

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='subject_name', help='The subject to process')
parser.add_argument('run', metavar='run_number', help='The run number of the subject')
args = parser.parse_args()
subject_name = args.subject
run = str(args.run)

print('Processing subject:', subject_name, 'run:', run)
plot_path = '../../plots/brain_gast/' + subject_name + '/' + subject_name+run
data_path = '../../derivatives/brain_gast/' + subject_name + '/' + subject_name+run
if not plot_path:
    os.makedirs(plot_path)
if not os.path.isdir(data_path):
    os.makedirs(data_path)

# loading the data
gastric_peak = np.load(data_path + '/max_freq' + subject_name + '_run' + run + clean_level + '.npy').flatten()
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
record_meta = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name) &
                                 (record_meta_pd['run'] == int(run)), :].to_dict('records')[0]
img_path = main_project_path + '/fmriprep/clean/{subject}/fmriprep/' \
                                 '{subject}_task-rest_run-0{run}_space-MNI152NLi' \
                                 'n6Asym_desc-preproc_bold_clean_' + clean_level + '.nii.gz'
img = nib.load(img_path.format(subject=subject_name, run=run))
vol = img.get_fdata()

if np.isclose(1/img.header.get_zooms()[3], sample_rate_fmri, rtol=0.0001, atol=0.0001):
    raise Exception ('TR length in the nifti header', img.header.get_zooms()[3], ' does not match the expected length', 1 /sample_rate_fmri)

# smoothing the mean image of the brain along the 300 time series.
smooth_image = image.smooth_img(img, fwhm=brain_fwhm)

# plot the smoothed image for quality control
fig2 = plt.figure(figsize=(28, 12))
mean_image = image.mean_img(smooth_image)
plotting.plot_img(mean_image, figure= fig2,colorbar = True, title = 'Mean Brain image after smoothing')
plt.savefig(plot_path + '/mean_brain2.png')
plt.close('all')

# now coverting the array to 2D- voxels and time
# including the voxels of the brain in the new array
# filtering the data & calculating hilbert transform of each relevant voxel.

mask = vol.std(axis=-1) > 0 # define a 3D mask
braindata = vol[mask, :] # convert the brain time series to a 2D array
if zscore_flag:
    braindata = scale(braindata,axis=1)

# plot the data before filtering for quality control
plt.plot(np.arange(braindata.shape[1]) * (1 / sample_rate_fmri), braindata.mean(axis=0))
plt.xlabel('Time (sec)'); plt.ylabel('Signal')
plt.savefig(plot_path + '/mask_vol.png')
plt.close('all')

braindata = filter_data(braindata , sfreq = sample_rate_fmri,
                        l_freq = gastric_peak - bandpass_lim, h_freq  = gastric_peak + bandpass_lim,
                        filter_length=int(filter_order*np.floor(sample_rate_fmri/(gastric_peak - bandpass_lim))),
                        l_trans_bandwidth=transition_width * (gastric_peak - bandpass_lim),
                        h_trans_bandwidth=transition_width * (gastric_peak + bandpass_lim),
                        n_jobs=1, method='fir', phase='zero-double', fir_window='hamming',
                        fir_design='firwin2')

plt.plot(np.arange(braindata.shape[1]) * (1 / sample_rate_fmri), braindata.mean(axis=0))
plt.xlabel('Time (sec)'); plt.ylabel('Signal')
plt.savefig(plot_path + '/filter_data_brain.png')
plt.close('all')

np.savez_compressed(data_path + '/func_filtered_' + subject_name + '_run' + run + clean_level + '.npz',
                    brain_signal = braindata)
np.savez_compressed(data_path + '/mask_' + subject_name + '_run' + run + clean_level + '.npz', mask = mask)
print('Done brain preprocessing for: ', subject_name, run)