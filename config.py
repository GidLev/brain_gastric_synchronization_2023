import os

#variables for anaylsis:
main_project_path = '/path/to/my/project/dir'
clean_level = 'strict' # 'strict' #'strict_gs' # 'strict_gs_cardiac' # 'minimal'
sample_rate_fmri = 0.5
intermediate_sample_rate = 10
trigger_channel = 8 # the channel number that recorded the trigger from the fMRI
filter_order = 5 # parameter related to the band pass filter
bandpass_lim = 0.015 # parameter related to the band pass filter (Hz)
transition_width = 15 / 100 # parameter related to the band pass filter, percent of half the filter size
extra_cut = 30 # cutting of time from the beginning and end of the signal, in seconds
window = 200 # welch window size
overlap = 100 # welch overlap in seconds
freq_range = [0.033,0.066] # normogastric range

multi_thread = True
num_threads = 8
brain_fwhm = 3 # spatial smooting of the brain
num_roi_permutations = 10000
ppu_sample_rate = 500
intermediate_ppu_sample_rate = 1
zscore_flag = True
subjects_null = False
single_run_flag = False

