
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

def get_curvature(sequence, points):
    from scipy.interpolate import interp1d
    from scipy.misc import derivative
    f = interp1d(np.arange(len(sequence)), sequence, kind='cubic')
    return [derivative(f, point, dx= 0.0001, n = 2) for point in points]

def powerspect(in_signal, window, overlap, res_freq, freq_range, plot_flag,
               subject_name, run, save_path, dominant='auto', dominant_freq = 'auto'):

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    power_mat = []
    for i in range(len(in_signal)):
        freq, power = signal.welch(in_signal[i], fs=res_freq, window='hanning', nperseg=window * res_freq,
                                    noverlap=overlap * res_freq, nfft=int(len(in_signal[i]))*3, scaling='spectrum')
        power_mat.append(power)
    a_idx, a = find_nearest(freq, freq_range[0])
    b_idx, b = find_nearest(freq, freq_range[1])
    freqcalc = freq[a_idx:b_idx]
    maxindex = []
    power_cut_list =[]
    max_power = []
    power_spect_data_list = []
    stats_print_short = ''
    if len(in_signal) > 1:
        spect_cross_chan_corr = np.corrcoef(np.array(power_mat)[:, a_idx:b_idx])
        spect_cross_chan_corr = spect_cross_chan_corr[~np.eye(spect_cross_chan_corr.shape[0],
                                                              dtype=bool)]
        spect_cross_chan_corr = spect_cross_chan_corr.reshape(spect_cross_chan_corr.shape[0],-1)
    for i in range(len(in_signal)):
        power_spect_data = {}
        powercut = power_mat[i][a_idx:b_idx]
        power_cut_list.append(powercut)
        peaks, peaks_properties = signal.find_peaks(powercut, prominence = [0,10],
                                                   width = [0,40])
        power_spect_data['channel'] = i
        if len(in_signal) > 1:
            power_spect_data['spect_cross_chan_corr'] = spect_cross_chan_corr[i, :].mean()
        else:
            power_spect_data['spect_cross_chan_corr'] = np.nan
        if len(peaks) == 0:
            print('No peaks were found for subject, run, electrode:', subject_name, run, i)
            maxindex.append(np.nan)
            max_power.append(np.nan)
            for key in ['curvature', 'height', 'frequency']:
                power_spect_data[key] = []
        else:
            peaks_properties['curvature'] = power_spect_data['curvature'] = get_curvature(powercut, peaks)
            peaks_properties['height'] = power_spect_data['height'] = powercut[peaks]
            peaks_properties['frequency'] = power_spect_data['frequency'] = freqcalc[peaks]
            max_peak = np.argmax(peaks_properties['height'])
            maxindex.append(freqcalc[peaks[max_peak]])
            max_power.append(peaks_properties['height'][max_peak])
            stats_print_short += 'Ch#' + str(i+1) + ': '
            stats_print = 'Channel #' + str(i+1) + ': '

            max_two_peaks = np.argsort(peaks_properties['height'])[-2:][::-1]
            for peak_i, peak in enumerate(max_two_peaks):
                stats_print_short += 'freq={:.3},h={:.2e},c={:.2e}'.format(peaks_properties['frequency'][peak],
                                                                                  peaks_properties['height'][peak],
                                                                                  peaks_properties['curvature'][peak])
                stats_print += 'frequency={:.4},height={:.3e},' \
                               'prominences={:.3e}, curvature={:.3e}'.format(peaks_properties['frequency'][peak],
                                                                             peaks_properties['height'][peak],
                                                                             peaks_properties['prominences'][peak],
                                                                             peaks_properties['curvature'][peak])
                if not peak_i == (len(max_two_peaks) - 1):
                    stats_print+= '; '
                    stats_print_short+= '; '
                elif not i == 3:
                    stats_print_short += '\n'
            print(stats_print)

        power_spect_data_list.append(power_spect_data.copy())
    if (dominant == 'auto') or (dominant == None):
        dominant_channel_num = np.where(max_power == np.nanmax(max_power))[0][0]
    else:
        dominant_channel_num = int(dominant)
    for i in range(len(power_spect_data_list)):
        if power_spect_data_list[i]['channel'] == dominant_channel_num:
            power_spect_data_list[i]['is_dominant'] = 1
        else:
            power_spect_data_list[i]['is_dominant'] = 0
   # dominant_channel_num = np.where(max(max_power))#maximum power between the channels, the dominant channel.

    if (dominant_freq == 'auto') or (dominant_freq == None):
        max_freq = maxindex[dominant_channel_num]
        auto_pick = True
    else:
        max_freq = dominant_freq
        auto_pick = False

    if plot_flag:
        egg_spectogram(freq, power_mat, stats_print_short, dominant_channel_num, max_freq,
                       freq_range, max_power[dominant_channel_num], save_path, subject_name, run)
        egg_spectogram_paper(freq, power_mat, stats_print_short, dominant_channel_num, max_freq,
                             freq_range, max_power[dominant_channel_num], save_path, subject_name, run,
                             auto_pick)
        roc_power = [np.mean(power_mat[i]) for i in range(len(in_signal))]
        roc_power_arg_sort = np.argsort(roc_power)
        if roc_power[roc_power_arg_sort[-1]] > (roc_power[roc_power_arg_sort[-1]] * 5):
            # if one channel has substantially more power than all other we also repeat the plot without it
            egg_spectogram(freq, power_mat, stats_print_short, dominant_channel_num, max_freq,
                           freq_range, max_power, save_path, subject_name, run,
                           chanel_mask=roc_power_arg_sort[-1])


    return max_freq, dominant_channel_num, power_spect_data_list

def egg_spectogram_paper(freq, power_mat, stats_print_short, dominant_channel_num, max_freq,
                   freq_range, max_power, save_path, subject_name, run, auto_pick, chanel_mask=None):
    colors_four_elect = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if auto_pick:
        selection_text = ''
    else:
        selection_text = 'Manually selected'
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i in range(len(power_mat)):
        if chanel_mask != None:
            ax.plot(freq, power_mat[i], label="Channel" + str(i + 1), color=colors_four_elect[i])
        elif chanel_mask != i:
            ax.plot(freq, power_mat[i], label="Channel" + str(i + 1), color=colors_four_elect[i])
    plt.legend(loc='upper right', fontsize=20)
    plt.xlabel('Frequency(Hz)', fontsize=22)
    plt.ylabel(r'$mV^{2}$', fontsize=22)  # r'power$(\frac{mv^{2}}{Hz})$'
    plt.xlim(0.01, 0.09)
    plt.ylim(0, max_power * 1.1)
    plt.plot(max_freq, max_power, marker="*", markersize=16)
    plt.annotate(selection_text,  # this is the text
                 (max_freq, max_power),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(4, 10),  # distance from text to points (x,y)
                 ha='right', fontsize=16)
    plt.vlines(freq_range[0], 0, max_power * 1.1, colors='k')
    plt.vlines(freq_range[1], 0, max_power * 1.1, colors='k')
    # plt.title(subject_name + " " + run, fontsize=20)
    plt.subplots_adjust(bottom=0.35, left=0.10)
    if chanel_mask == None:
        plt.savefig(save_path + '/paper_egg_power_spectral_density_' + subject_name + '_' + run + '.png')
    else:
        plt.savefig(save_path + '/paper_egg_power_spectral_density_' + subject_name + '_' + run + '_masked.png')
    plt.close('all')


def egg_spectogram(freq, power_mat, stats_print_short, dominant_channel_num, max_freq,
                   freq_range, max_power, save_path, subject_name, run, chanel_mask = None):
        domin_name = 'Channel' + str((dominant_channel_num + 1))
        colors_four_elect = [[1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0.222, 1]]
        fig, ax = plt.subplots(figsize=(15, 6))
        for i in range(len(power_mat)):
            if chanel_mask != None:
                ax.plot(freq, power_mat[i], label="channel"+str(i+1), color = colors_four_elect[i])
            elif chanel_mask != i:
                ax.plot(freq, power_mat[i], label="channel"+str(i+1), color = colors_four_elect[i])
        plt.legend(loc='upper right',fontsize = 'xx-large')
        plt.xlabel('Frequency(Hz) \n\n' + stats_print_short, fontsize=18)
        plt.ylabel(r'$mV^{2}$', fontsize=18) #r'power$(\frac{mv^{2}}{Hz})$'
        plt.xlim(0.01,0.09)
        plt.ylim(0,max_power*1.1)
        plt.plot(max_freq, max_power, marker="*", markersize=10)
        plt.vlines(freq_range[0], 0, max_power*1.1, colors = 'k')
        plt.vlines(freq_range[1], 0, max_power*1.1, colors = 'k')
        # plt.title(subject_name + " " + run, fontsize=20)
        plt.title('EGG power spectral density', fontsize=18)
        plt.annotate(domin_name,  # this is the text
                    (max_freq, max_power),  # this is the point to label
                    textcoords="offset points",  # how to position the text
                    xytext=(4, 10),  # distance from text to points (x,y)
                    ha='right',fontsize=16)
        plt.subplots_adjust(bottom=0.35, left=0.10)
        if chanel_mask == None:
            plt.savefig(save_path + '/egg_power_spectral_density_' + subject_name + '_' + run +'.png')
        else:
            plt.savefig(save_path + '/egg_power_spectral_density_' + subject_name + '_' + run +'_masked.png')
        plt.close('all')


