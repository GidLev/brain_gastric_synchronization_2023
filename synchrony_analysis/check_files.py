import pandas as pd
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import clean_level, main_project_path

# get the data
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data.csv')

if clean_level == 'strict_gs_cardiac':
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]

for run_indx in record_meta_pd.index.values:
    data_paths = []
    subject_name = record_meta_pd.loc[run_indx,'subject']
    run = str(record_meta_pd.loc[run_indx,'run'])
    data_paths.append(f'{main_project_path}/fmriprep/out/{subject_name}/fmriprep/{subject_name}_task-rest_run-0{run}'
                      f'_space-MNI152NLin6Asym_desc-preproc_bold_{clean_level}.nii.gz')
    data_paths.append(f'{main_project_path}/physio/{subject_name}/egg/{subject_name}_rest{run}.acq')
    if clean_level == 'strict_gs_cardiac':
        data_paths.append(f'{main_project_path}/physio/{subject_name}/ppu/{subject_name}_rest{run}.log')
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'file not found: {data_path}')

print('Done files pre check.')


