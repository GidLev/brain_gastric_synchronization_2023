import bioread
import pandas as pd
import scipy as sp
import numpy as np
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from utils.gastric_utils import plot_signal, plot_trigger
from utils.spect_utils import powerspect
from mne.filter import filter_data
from scipy import stats
import argparse
import os
from config import main_project_path, intermediate_sample_rate, \
    window, overlap, freq_range, bandpass_lim, transition_width, \
    filter_order, zscore_flag, clean_level, trigger_channel
import warnings

# # Handle command line arguments
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

#load the data
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
record_meta = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name) &
                                    (record_meta_pd['run'] == int(run)),:].to_dict('records')[0]
data = bioread.read_file(main_project_path + '/physio/{subject}/egg/{subject}_rest{run}.acq')

# Variable definitions
original_sample_rate = data.channels[0].samples_per_second # sample rate of the signal
signal_time = data.channels[0].time_index # the time label (in seconds) for each time point in the signal
duration = original_sample_rate * record_meta['mri_length'] # the expected duration of the data in seconds
num_gast = record_meta['num_channles'] # number of gastric channels

## Get the MRI trigger indecies or calculate them
trigger = data.channels[trigger_channel].data  # data from trigger signal (change `trigger_channel` to the appropriate channel number)
trigger = trigger.astype(int)
if record_meta['trigger_start'] == 'auto':
    if trigger[0] == 0:
        action_idx = [np.where(trigger)[0][0], np.where(trigger)[0][0] + int(duration)]
    else:
        no_trigger_locs = np.where(trigger == 0)[0]
        trigger_locs = np.where(trigger >= 0.999)[0]
        str_loc = trigger_locs[trigger_locs > no_trigger_locs[0]][0]
        action_idx = [str_loc, str_loc + int(duration)]
        warnings.warn('The signal started with the trigger channel larger then zero, skipped to the next one')
else:
    print('No auto Trigger Signal exist')
    trigger_start = int(max(float(record_meta['trigger_start']),0) * original_sample_rate)
    action_idx = [trigger_start, trigger_start + int(duration)]

plot_trigger(trigger, action_idx, original_sample_rate, subject=subject_name,
            run=run, save_path=plot_path)

# Slicing the recording according to the MRI trigger
signal_time = data.channels[0].time_index[:int(duration)]
signal_egg = [data.channels[i].data[action_idx[0]:action_idx[1]] for i in range(num_gast)]

plot_signal(signal_time, signal_egg, 'EGG sliced signal',
            'sliced_signal', subject_name + '_' + run, plot_path)


# Resampling the data to 10Hz
resample_n_points = int((len(signal_time) / original_sample_rate) * intermediate_sample_rate) #after we cut the signal
signal_egg = [sp.signal.resample(signal_egg[i], resample_n_points) for i in range(len(signal_egg))]
signal_time = np.arange(0, resample_n_points / intermediate_sample_rate, 1.0 / intermediate_sample_rate)
plot_signal(signal_time, signal_egg, 'EGG signal after resampling',
            'post_first_resample_', subject_name + '_' + run, plot_path)

# Welch & peak channel and frequency
max_freq, dominant_channel_num, power_spect_data_list = powerspect(signal_egg, window, overlap, intermediate_sample_rate,
                                                       freq_range, True, subject_name, run, plot_path,
                                                       dominant = record_meta['dominant_channel'],
                                                       dominant_freq = record_meta['dominant_frequency'])

if (record_meta['dominant_channel'] != 'auto') | \
        (record_meta['dominant_frequency'] != 'auto'):
    print('Manual channel set, overriding')
    if (record_meta['dominant_channel'] == 'auto') & \
            (record_meta['dominant_frequency'] == 'auto'):
        raise Exception ('Both variables "dominant_channel" and "dominant_frequency" must be set if used')
    max_freq = float(record_meta['dominant_frequency'])
    dominant_channel_num = int(record_meta['dominant_channel'])


signal_egg_selected = filter_data(signal_egg[dominant_channel_num], sfreq=intermediate_sample_rate,
                                  l_freq=max_freq - bandpass_lim, h_freq=max_freq + bandpass_lim,
                                  picks=None, n_jobs=1, method='fir', phase='zero-double',
                                  filter_length=int(filter_order*np.floor(intermediate_sample_rate/(max_freq - bandpass_lim))),
                                  l_trans_bandwidth=transition_width * (max_freq - bandpass_lim),
                                  h_trans_bandwidth=transition_width * (max_freq + bandpass_lim),
                                  fir_window='hamming', fir_design='firwin2')

if zscore_flag:
    signal_egg_selected = stats.zscore(signal_egg_selected)

plot_signal(signal_time, [signal_egg_selected], 'EGG filtered',
            'egg_filtered', subject_name + '_' + run, plot_path)

np.save(data_path + '/gast_data_' + subject_name + '_run' + run + clean_level + '.npy', signal_egg_selected)
np.save(data_path + '/max_freq' + subject_name + '_run' + run + clean_level + '.npy', max_freq)
print('Done gastric preprocessing for: ', subject_name, run)