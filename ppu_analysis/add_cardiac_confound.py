'''
The script adds cardiac confound to fmriprep confounds file based on niphlem RETROICOR method.
This is a more simplistic approach than the one used in the FSL NPM pipeline (done whole brain and not per slice).
source: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/NPM/UserGuide
        https://github.com/CoAxLab/niphlem/blob/main/niphlem/models.py
        https://coaxlab.github.io/niphlem/

references: https://doi.org/10.1002/1522-2594(200007)44:1<162::AID-MRM23>3.0.CO;2-E

'''

import pandas as pd
import os
import sys
import pathlib
import nibabel as nib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import main_project_path, ppu_sample_rate
import ppu_utils


def retroicor_regressors(epi_file, ppu_signal, ppu_sr, order, ppu_peaks):
    """
    Generate a set of regressors for the retroicor model.

        First, peaks in the signal are identified. Then, phases
        are sampled such a complete cycle [0, 2pi] happens between peaks.
        Then, phases are expanded up to or for a given fourier order. Then,
        phases are sub-sampled to the scanner time. Finally, sine and cosine
        are generated for each fourier mode.

    Parameters
    ----------
    epi_file : str
        Path to the EPI file.
    ppu_signal : ndarray
        The PPU signal (PPG signal).
    ppu_sr : int
        The sampling rate of the ppu signal.
    order: int
        Fourier expansion for phases. The fourier expansion is
        performed to that order, starting from 1.
    ppu_peaks : list
        List of peaks in the PPU signal.

    Returns
    -------
    physi_regressors : ndarray
        The physiological regressors.

    """
    import numpy as np
    import nibabel as nib
    from scipy.interpolate import interp1d
    # convert peaks from bool to indexes if needed
    if ppu_peaks.dtype == bool:
        ppu_peaks = np.where(ppu_peaks)[0]

    # Get the number of volumes in the EPI file.
    n_volumes = nib.load(epi_file).shape[3]
    TR = nib.load(epi_file).header.get_zooms()[3]

    # create time tick (in seconds) for the ppu and brain scan
    time_scan = np.arange(0, n_volumes * TR, TR)
    time_physio = np.arange(0, len(ppu_signal) / ppu_sr, 1.0 / ppu_sr)

    # Compute phases according to peaks (changed to an interpolation)
    phases_in_peaks = 2 * np.pi * np.arange(len(ppu_peaks))
    phases = interp1d(x=time_physio[ppu_peaks],
                      y=phases_in_peaks,
                      kind="linear",
                      fill_value="extrapolate")(time_physio)

    # Expand phases according to Fourier expansion order
    phases_fourier = [(m * phases).reshape(-1, 1)
                      for m in range(1, int(order) + 1)]
    phases_fourier = np.column_stack(phases_fourier)

    # Subsample phases to the scanner time
    phases_scan = np.zeros((len(time_scan), phases_fourier.shape[1]))
    for ii, phases in enumerate(phases_fourier.T):
        interp = interp1d(time_physio,
                          phases,
                          kind='linear',
                          fill_value='extrapolate')
        phases_scan[:, ii] = interp(time_scan)

    # Apply sin and cos functions
    sin_phases = np.sin(phases_scan)
    cos_phases = np.cos(phases_scan)

    # This is just to be ordered according to the fourier expansion
    regressors = [np.column_stack((a, b))
                  for a, b in zip(sin_phases.T, cos_phases.T)]
    return np.column_stack(regressors)


def read_ppu_signal(ppu_file, mri_length):
    """
    Read the PPU signal from a file.

    Parameters
    ----------
    ppu_file : str
        Path to the PPU signal file.

    Returns
    -------
    ppu_signal : ndarray
        The PPU signal.
    """
    data = ppu_utils.ppu_parse_log_file(ppu_file).iloc[:, 1:]
    column_names = data.columns.tolist()
    data = data.values[-1 * mri_length:, :]
    return data[:, column_names.index('ppu')]


# read the subject list
rewrite = False
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]  # filter subjects without oximeter data
fmriprep_dir = main_project_path + '/fmriprep/out/out/fmriprep'
fmri_file_template = fmriprep_dir + '/{subject}/func/{subject}_task-rest_run-' \
                                     '0{run}_space-MNI152NLin6Asym_desc-preproc_' \
                                     'bold.nii.gz'
confounds_template = fmriprep_dir + '/{subject}/func/{subject}_task-rest_run-' \
                                    '0{run}_desc-confounds_regressors.tsv'
ppu_template = main_project_path + '/physio/{subject}/ppu/{subject}_rest{run}.log'

templates = {'fmri': fmri_file_template, 'confounds': confounds_template, 'ppu': ppu_template}

for subject, run in record_meta_pd.loc[:, ['subject', 'run']].values:
    print('processing ', subject, run)
    files_found = True
    for template in templates.keys():
        file_path = templates[template].format(subject=subject, run=run)
        if not os.path.exists(file_path):
            print('file not found: ', file_path)
            files_found = False
    confounds_file = confounds_template.format(subject=subject, run=run)
    if files_found:
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        TR = nib.load(fmri_file_template.format(subject=subject, run=run)).header.get_zooms()[-1]
        ppu_signal = read_ppu_signal(ppu_template.format(subject=subject, run=run),
                                     mri_length=int(confounds_df.shape[0] * TR * ppu_sample_rate))
        ppu_peaks = ppu_utils.oxi_peaks(ppu_signal, sfreq=ppu_sample_rate)
        ppu_confounds = retroicor_regressors(epi_file=fmri_file_template.format(subject=subject, run=run),
                                             ppu_signal=ppu_signal,
                                             ppu_sr=ppu_sample_rate, order=3,
                                             ppu_peaks=ppu_peaks)
        ppu_confounds = pd.DataFrame(ppu_confounds, columns=[
            'ppu_comp_%d' % i for i in range(ppu_confounds.shape[1])])
        confounds_df = pd.concat([confounds_df, pd.DataFrame(ppu_confounds)], axis=1)
        confounds_df.to_csv(confounds_file, index=False,sep='\t')
        print('saved confounds file: ', confounds_file)
    else:
        print('File not found or confound file already created:', subject, run)
