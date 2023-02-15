"""
https://github.com/aaltoimaginglanguage/study_template
Van Vliet, M. (2020). Seven quick tips for analysis scripts in neuroimaging.
PLoS computational biology, 16(3), e1007358.
"""

import subprocess
import sys
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from config import num_threads, multi_thread, clean_level

def call(params):
    """Call a script and exit if the script failed."""
    script = params[0]
    script_params = params[1:]
    return_code = subprocess.call(['python', script] + script_params)
    if return_code != 0:
        sys.exit(return_code)

# read and filter the subjects list
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')
if clean_level == 'strict_gs_cardiac':
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]

subprocess.call(['python', 'brain_analysis/run_fmriprep.py'])
if clean_level == 'strict_gs_cardiac':
    subprocess.call(['python', 'ppu_analysis/add_cardiac_confound.py'])
subprocess.call(['python', 'brain_analysis/fmri_cleaning.py'])
subprocess.call(['python', 'synchrony_analysis/check_files.py'])

# Do all analysis steps for all subjects with [num_threads] workers
processing_scripts = ['egg_analysis/preprocess_gastric.py',
                      'brain_analysis/brain_preprocessing.py',
                      'synchrony_analysis/signal_slicing.py',
                      'synchrony_analysis/voxel_based_analysis.py',
                     ]

for processing_script in processing_scripts:
    args_list = []
    for run_indx in record_meta_pd.index.values:
        subject_name = record_meta_pd.loc[run_indx, 'subject']
        run = str(record_meta_pd.loc[run_indx, 'run'])
        args_list.append([processing_script,
                          record_meta_pd.loc[run_indx,'subject'],
                          str(record_meta_pd.loc[run_indx,'run'])])
    if multi_thread:
        pool = ThreadPool(num_threads)
        pool.map(call, args_list)
    else:
        for args in args_list:
            call(args)

# Perform the group analysis
subprocess.call(['python', 'synchrony_analysis/voxel_based_second_level.py'])

print('Master script done.')
