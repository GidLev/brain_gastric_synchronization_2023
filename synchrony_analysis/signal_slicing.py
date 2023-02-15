import numpy as np
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import sample_rate_fmri, extra_cut, intermediate_sample_rate, clean_level
import pandas as pd
import argparse

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='subject_name', help='The subject to process')
parser.add_argument('run', metavar='run_number', help='The run number of the subject')
args = parser.parse_args()
subject_name = args.subject
run = str(args.run)

record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
record_meta = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name) &
                                    (record_meta_pd['run'] == int(run)),:].to_dict('records')[0]
data_path = '../../derivatives/brain_gast/' + subject_name + '/' + subject_name+run

gastric_signal = np.load(data_path + '/gast_data_' + subject_name + '_run' + run + clean_level + '.npy')
brain_signal = np.load(data_path + '/func_filtered_' + subject_name + '_run' + run + clean_level + '.npz')['brain_signal']
gastric_sr_ratio = sample_rate_fmri / intermediate_sample_rate

# cut the signals 1. to match each other 2. to remove edge effect
if record_meta['trigger_start'] != 'auto':
    if float(record_meta['trigger_start']) < 0:
        print('Negative trigger value, slicing the fMRI time-series')
        brain_signal = brain_signal[:,int(-1*int(record_meta['trigger_start'])*sample_rate_fmri):]
if (gastric_signal.shape[0]*gastric_sr_ratio) !=  brain_signal.shape[-1]:
    print('the brain and gastric signals dont match, slicing them to the minimum length')
    min_length_mri = min(int(gastric_signal.shape[0]*gastric_sr_ratio), brain_signal.shape[1])
    min_length_egg = int(min_length_mri * (1/gastric_sr_ratio))
    gastric_signal = gastric_signal[int(intermediate_sample_rate*extra_cut):min_length_egg - int(intermediate_sample_rate*extra_cut)]
    brain_signal = brain_signal[:,int(sample_rate_fmri*extra_cut):min_length_mri - int(sample_rate_fmri*extra_cut)]
else:
    gastric_signal = gastric_signal[int(intermediate_sample_rate*extra_cut):-1*int(intermediate_sample_rate*extra_cut)]
    brain_signal = brain_signal[:,int(sample_rate_fmri*extra_cut):-1*int(sample_rate_fmri*extra_cut)]
print('signal length: ' + str(brain_signal.shape[1]/sample_rate_fmri) + ' sec')
np.save(data_path + '/gast_data_' + subject_name + '_run' + run + clean_level + '_sliced.npy', gastric_signal)
np.savez_compressed(data_path + '/func_filtered_' + subject_name + '_run' + run + clean_level + '_sliced.npz',
                    brain_signal=brain_signal)
