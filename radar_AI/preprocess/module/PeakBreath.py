import numpy as np


def peakbreath(data_in):
    breath_start_freq_index = 5  
    breath_end_freq_index = 20  
    
    p_peak_values = np.zeros(128) 
    p_peak_index = np.zeros(128) 
    max_num_peaks_spectrum = 4  
    num_peaks = 0

    for i in range(breath_start_freq_index, breath_end_freq_index):
        if data_in[i] > data_in[i - 1] and data_in[i] > data_in[i + 1]:
            p_peak_index[num_peaks] = i
            p_peak_values[num_peaks] = data_in[i]
            num_peaks += 1
    if num_peaks < max_num_peaks_spectrum:
        index_num_peaks = num_peaks
    else:
        index_num_peaks = max_num_peaks_spectrum

    p_peak_index_sorted = np.zeros(index_num_peaks)
    if index_num_peaks != 0:
        for i in range(index_num_peaks):
            idx = np.argmax(p_peak_values)
            p_peak_index_sorted[i] = idx
            p_peak_values[idx] = 0
        max_index_breath_spect = p_peak_index[int(p_peak_index_sorted[0])]
    else:
        max_index_breath_spect = np.argmax(data_in[breath_start_freq_index:breath_end_freq_index])

    res = 60.0 * (max_index_breath_spect - 1) * 0.0195
    res_index = max_index_breath_spect

    return res, res_index
