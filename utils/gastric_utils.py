from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

def plot_signal(x_axis, data, main_title, prefix,id, save_path):
    num_chan = len(data)
    plt.figure(figsize=(30, 12))
    for chan in range(len(data)):
        plt.subplot(num_chan, 1, chan + 1)
        plt.plot(x_axis, data[chan])
    plt.xlabel('Time(sec)', fontsize=10)
    plt.ylabel('Voltage (mv)', fontsize=10)
    plt.title("Channel number " + str(chan+1),fontsize=11)
    plt.tight_layout(pad=6.0, w_pad=6.0, h_pad=6.0)
    plt.suptitle(main_title,fontsize=18)
    plt.savefig(save_path + '/' + prefix + id +'.png')
    print('a --- figure was saved to:', save_path)
    plt.close('all')

def roundup(x, to):
    return x if x % to == 0 else x + to - x % to

def resampling(x_axis, data, num_iter, subtitle, rows, columns, resampling,main_title,plot_flag,
               id,save_path):
    y = []
    if plot_flag:
        plt.figure(figsize=(30, 12))
    for chan in range(num_iter):
        y.append(signal.resample(data[chan], resampling))
        if plot_flag:
            plt.subplot(rows, columns, chan + 1)
            plt.plot(x_axis, y[chan])
            plt.xlabel('Time(sec)', fontsize=10)
            plt.ylabel('Voltage (mv)', fontsize=10)
            try:
                max_x_label_value = roundup(int(x_axis[-1]), 100)
            except:
                max_x_label_value = 600
            plt.xlim(0,max_x_label_value)
            plt.title(subtitle + " " + str(chan + 1),fontsize=11)
            plt.tight_layout(pad=6.0, w_pad=6.0, h_pad=6.0)
            plt.suptitle(main_title,fontsize=18)
    plt.savefig(save_path + '/Resampling_10Hz' + id + '.png')
    plt.close('all')
    return y


def plot_trigger(trigger, action_idx, sample_rate, subject = '', run = '', save_path = None, show=False):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.plot(np.arange(0,len(trigger)/ sample_rate,1/sample_rate), trigger, c='b')
    plt.axvline(x=action_idx[0]/ sample_rate, c='g', label = 'sample start')
    plt.axvline(x=action_idx[1] / sample_rate, c='r', label = 'sample end')
    plt.legend(loc='upper right')
    if save_path != None:
        plt.savefig(save_path + '/trigger_cut_' + subject + '_' + run +'.png')
    if show:
        plt.show()
    plt.close('all')

def to_phase_resampled(signal_1d, origin_sr, target_sr):
    import scipy as sp
    signal_1d = signal.hilbert(signal_1d)
    signal_1d = np.angle(signal_1d)
    ## Resampeling to the brain sampling rate (0.5)
    resample_n_points = int(
        (len(signal_1d) / origin_sr) * target_sr)  # Number of samples after resampling
    return sp.signal.resample(signal_1d, resample_n_points)

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import (sample_rate_fmri, intermediate_sample_rate)
import scipy as sp
from sklearn.preprocessing import scale


def plot_example_synchrony(gastric_signal, brain_signal, plvs_empirical, plot_path):
    resample_n_points = int(
        (len(gastric_signal) / intermediate_sample_rate) * sample_rate_fmri)  # Number of samples after resampling
    gastric_signal_resamp =  sp.signal.resample(gastric_signal, resample_n_points)
    example_channels = [plvs_empirical.argmax(), np.random.randint(len(plvs_empirical)), plvs_empirical.argmin()] #np.argsort(plvs_empirical)[-1*(len(plvs_empirical) / 2)]
    example_names = ['Voxel with high synchrony (PLV={:.2f})'.format(plvs_empirical[example_channels[0]]),
                     'Randomly selected voxel (PLV={:.2f})'.format(plvs_empirical[example_channels[1]]),
                     'Voxel with low synchrony (PLV={:.2f})'.format(plvs_empirical[example_channels[2]])]
    fig, axes = plt.subplots(len(example_channels), figsize=(12,9))
    fig.suptitle('Example BOLD-gastric synchrony')
    for i in np.arange(len(example_channels)):
        axes[i].plot(np.arange(len(gastric_signal_resamp)) / sample_rate_fmri,
                     scale(brain_signal[example_channels[i],:]), label='BOLD signal')
        axes[i].plot(np.arange(len(gastric_signal_resamp)) / sample_rate_fmri,
                     scale(gastric_signal_resamp), label='Gastric signal')
        axes[i].set_title(example_names[i])
        axes[i].set_ylabel('Signal amplitude')
        axes[i].set_xlabel('Time(sec)')
        axes[i].set_xlim(0, 240)
        axes[i].set_xticks(np.arange(0,240,60))
        axes[i].grid()
    plt.xlabel('Time (sec)')
    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.4), ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(plot_path, dpi=200)
    plt.close('all')