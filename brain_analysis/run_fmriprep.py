from datetime import datetime
import time
import warnings
import docker
import sys
import os
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import main_project_path

def build_subject_list(bids_dir, log_dirs):
    """ Build a list of subjects to run fmriprep on """
    subject_list = [x for x in os.listdir(bids_dir) if ('sub-' in x)]
    for log_dir in log_dirs:
        subject_list = [x for x in subject_list if
                        (not os.path.isfile(log_dir + '/docker_report_' + x + '.txt'))]
    return subject_list

def call(bids_dir, output_dir, tmp_dir, fs_dir, fs_license, fmriprep_v,
         extra_options, subjects_l, log_dirs):
    '''
    Perform a docker run command, the input are assigned as follows:
    docker run -ti --rm \
        -v [bids_dir]:/data:ro \
        -v [output_dir]:/out \
        -v [tmp_dir]:/tmp \
        -v [fs_fs_license]:/opt/freesurfer/license.txt \
        [fmriprep_v] \
        /data /out/out \
        participant --participant_label [subject]\
        -w /tmp [extra_options]

    Example:
        docker run -ti --rm \
            -v /media/data2/EGG_rest/Subjects_EGG_fMRI/BIDS/soroka:/data:ro \
            -v /media/data2/EGG_rest/Subjects_EGG_fMRI/fmriprep/out:/out \
            -v /media/data2/EGG_rest/Subjects_EGG_fMRI/fmriprep/tmp:/tmp \
            -v /usr/local/freesurfer/license.txt:/opt/freesurfer/license.txt \
            nipreps/fmriprep:20.0.6 \
            /data /out/out \
            participant --participant_label sub-CERB\
            -w /tmp --fs-subjects-dir /freesurfer --nthreads 6 --omp-nthreads 6 --output-spaces MNI152NLin6Asym anat --fs-license-file /opt/freesurfer/license.txt --notrack --bold2t1w-dof 6
    '''

    command = ['/data', '/out/out',
               'participant', '--participant_label'] + subjects_l + \
              ['-w', '/tmp', '--fs-subjects-dir', '/freesurfer'] + extra_options
    volumes = [bids_dir + ':/data:ro', output_dir + ':/out',
               tmp_dir + ':/tmp',
               fs_dir + ':/freesurfer',
               fs_license + ':/opt/freesurfer/license.txt']
    for log_dir in log_dirs:
        script_path = log_dir + '/docker_command_' + " ".join(subjects_l) + '.txt'
        with open(script_path, 'w') as text_file:
            text_file.write(" ".join(command))
    print('Running ' + " ".join(command) + ' current time: ' +
          str(datetime.now()))
    try:
        t_start = time.perf_counter()
        client = docker.from_env()
        log = client.containers.run(fmriprep_v, command, volumes=volumes,
                                    tty=True, auto_remove=True)
        # for debugging use auto_remove=False and remove=False, pause after fail and print log
        elapsed_time = time.perf_counter() - t_start
        print(" ".join(subjects_l) + 'finished successfully in ', elapsed_time, 'second.')
        for subject in subjects_l:
            for log_dir in log_dirs:
                final_report_path = log_dir + '/docker_report_' + subject + '.txt'
                with open(final_report_path, "w") as text_file:
                    text_file.write('docker finished in :  ' + str(datetime.now()))
                    text_file.write('Took  ' + str(elapsed_time) + ' second.')
    except:
        warnings.warn('Docker failed time: '  + str(datetime.now()) + ' (delete the resulting log file to try again)')

bids_dir = main_project_path + '/BIDS_data/soroka'
output_dir = main_project_path + '/fmriprep/out'
tmp_dir = main_project_path + '/fmriprep/tmp'
fs_license = '/usr/local/freesurfer/license.txt'
fs_dir = main_project_path + '/fmriprep/freesurfer'
log_dir = main_project_path + '/fmriprep/run_logs'

fmriprep_v = 'nipreps/fmriprep:20.0.6'
extra_options = ['--nthreads', '6', '--omp-nthreads', '6',
                 '--output-spaces', 'MNI152NLin6Asym', 'anat',
                 '--fs-license-file', '/opt/freesurfer/license.txt',
                 '--notrack', '--skip-bids-validation', '--bold2t1w-dof', '6']

subs_in_parallel = 1

subject_list = build_subject_list(bids_dir, [log_dir])

print('Start running fmriprep docker')

while len(subject_list) > 0:
    print(str(len(subject_list)) + ' subjects left.')
    if subs_in_parallel == 1:
        subjects_l = [subject_list[0]]
    else:
        subjects_l = subject_list[:subs_in_parallel]
    call(bids_dir, output_dir, tmp_dir, fs_dir, fs_license, fmriprep_v,
         extra_options, subjects_l, [log_dir])
    subject_list = build_subject_list(bids_dir, [log_dir])

print('Done running fmriprep docker.')

