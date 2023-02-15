import os
import pandas as pd
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
import nibabel as nib
from nilearn.input_data import NiftiMasker
from multiprocessing.dummy import Pool as ThreadPool
from config import main_project_path

def clean_func(inputs, confounds):
    """Clean the functional data by regressing out the nuisance regressors specified in [confounds]."""
    # read the subject's functional data
    try:
        subject, run = inputs
        confounds_df = pd.read_csv(confounds_template.
                                   format(subject=subject, run=run), sep='\t')
        rest_file = rest_file_template.format(subject=subject, run=run)
        mask_file = mask_template.format(subject=subject, run=run)
        output_path = output_file_template.format(subject=subject[4:], run=run)

        # clean confounds - see https://github.com/simexp/load_confounds
        confounds_df = dropna_confounds(confounds_df.loc[:,confounds])
        rest_img = nib.load(rest_file)
        t_r = rest_img.header.get_zooms()[3]
        masker = NiftiMasker(mask_img = mask_file,
                             standardize='zscore', low_pass=0.1, high_pass=0.01,
                             detrend=True, t_r = t_r)
        clean_in_mask = masker.fit_transform(rest_file, confounds=confounds_df.values)
        clean_img = masker.inverse_transform(clean_in_mask)
        clean_img = nib.Nifti1Image(clean_img.get_fdata(), affine=rest_img.affine,
                                    header=rest_img.header)
        # save to nii gz file
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        nib.save(clean_img, output_path)
        print('Clean image successfully created for subject', subject)
    except Exception as e:
        print('Error occurred while processing subject', subject,
               'run', run, ':\n', e)

def dropna_confounds(df):
    '''
    Deal with NaNs in first time point within the confounds dataframe
    https://github.com/poldracklab/fitlins/issues/56
    '''
    for motion_col in ['framewise_displacement', 'dvars']:
        if motion_col in df.columns:
            df.loc[0,motion_col] = 0
    df.loc[0, :] = df.loc[0, :].fillna(0)
    return df

def get_confounds(confounds = ['csf'], n_comp_cor = 0):
    '''
    Get a list of confounds to be regressed out from the confounds file.
    The list is based on the  [confounds] parameter and the number of [n_comp_cor] specified.
   '''
    if n_comp_cor > 0:
        confounds += [f"a_comp_cor_{c:02d}" for c in range(n_comp_cor)]
    return confounds

multithreading = True
clean_level = 'strict' # 'strict' #'strict_gs' # 'strict_gs_cardiac' # 'minimal'
n_processes = 6
overwrite = True
fmriprep_dir = main_project_path + '/fmriprep/out/out/fmriprep'
rest_file_template = fmriprep_dir + '/{subject}/func/{subject}_task-rest_run-' \
                                     '0{run}_space-MNI152NLin6Asym_desc-preproc_' \
                                     'bold.nii.gz'
confounds_template = fmriprep_dir + '/{subject}/func/{subject}_task-rest_run-' \
                                    '0{run}_desc-confounds_regressors.tsv'
mask_template = fmriprep_dir + '/{subject}/func/{subject}_task-rest_run-' \
                               '0{run}_space-MNI152NLin6Asym_desc-brain_' \
                               'mask.nii.gz'
output_file_template = main_project_path + '/fmriprep/clean/{subject}/fmriprep/' \
                                 '{subject}_task-rest_run-0{run}_space-MNI152NLi' \
                                 'n6Asym_desc-preproc_bold_clean_' + clean_level + '.nii.gz'
bids_dir = main_project_path + '/BIDS_data/soroka'
if clean_level == 'minimal':
    confounds = get_confounds(n_comp_cor = 0)
elif clean_level == 'strict_gs':
    confounds_l = [
        'global_signal',
        'trans_x', 'trans_x_derivative1', 'trans_x_power2',
        'trans_y', 'trans_y_derivative1', 'trans_y_power2',
        'trans_z', 'trans_z_derivative1', 'trans_z_power2',
        'rot_x', 'rot_x_derivative1', 'rot_x_power2',
        'rot_y', 'rot_y_derivative1', 'rot_y_power2',
        'rot_z', 'rot_z_derivative1', 'rot_z_power2',
    ]
    confounds = get_confounds(confounds_l, n_comp_cor = 6)# define list of confounds
elif clean_level == 'strict':
    confounds_l = [
        'trans_x', 'trans_x_derivative1', 'trans_x_power2',
        'trans_y', 'trans_y_derivative1', 'trans_y_power2',
        'trans_z', 'trans_z_derivative1', 'trans_z_power2',
        'rot_x', 'rot_x_derivative1', 'rot_x_power2',
        'rot_y', 'rot_y_derivative1', 'rot_y_power2',
        'rot_z', 'rot_z_derivative1', 'rot_z_power2',
    ]
    confounds = get_confounds(confounds_l, n_comp_cor = 6)# define list of confounds
elif clean_level == 'strict_gs_cardiac':
    confounds_l = [
        'global_signal',
        'trans_x', 'trans_x_derivative1', 'trans_x_power2',
        'trans_y', 'trans_y_derivative1', 'trans_y_power2',
        'trans_z', 'trans_z_derivative1', 'trans_z_power2',
        'rot_x', 'rot_x_derivative1', 'rot_x_power2',
        'rot_y', 'rot_y_derivative1', 'rot_y_power2',
        'rot_z', 'rot_z_derivative1', 'rot_z_power2',
        'ppu_comp_0', 'ppu_comp_1', 'ppu_comp_2',
        'ppu_comp_3', 'ppu_comp_4', 'ppu_comp_5',
    ]
    confounds = get_confounds(confounds_l, n_comp_cor = 6)# define list of confounds

subject_list =  [x for x in os.listdir(bids_dir) if ('sub-' in x)]

tasks_list = []

for subject in subject_list:
    for run in [1,2,3]:
        if os.path.isfile(rest_file_template.format(subject=subject, run=run)) and \
           os.path.isfile(confounds_template.format(subject=subject[4:], run=run)) and  \
           os.path.isfile(mask_template.format(subject=subject, run=run)) and  \
           (not os.path.isfile(output_file_template.format(subject=subject[4:], run=run)) or overwrite):
            tasks_list.append((subject, run))

subject_l, run_l = [], []
for subject, ses, run in tasks_list:
    subject_l.append(subject[4:])
    run_l.append('{ses}_0{run}'.format(run=run, ses=ses))

if multithreading:
    pool = ThreadPool(n_processes)
    results = pool.map(clean_func, tasks_list)
else:
    for subject, run in tasks_list:
        clean_func((subject, run))

print('Done')


