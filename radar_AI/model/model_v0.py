import numpy as np
from scipy.signal import windows, butter, sosfilt
import math

def model_v0(data):
    # FFT 스펙트럼 부분
    [numchirps, chirpsamples] = np.shape(data)
    avgs = np.average(data, 1).reshape(numchirps, 1)
    data = data - avgs
    range_window = windows.blackmanharris(256).reshape(1, 256)
    data = np.multiply(data, range_window)
    zp1 = np.pad(data, ((0, 0), (0, chirpsamples)), 'constant')
    range_fft = np.fft.fft(zp1) / chirpsamples
    range_fft = 2 * range_fft[:, range(int(chirpsamples))]

    # 피크 추출 부분
    start_index = 40
    end_index = 80
    num_frame = range_fft.shape[0]
    result = np.zeros(num_frame, dtype=complex)
    phase = np.zeros(num_frame)

    for frame_index in range(num_frame):
        maxvalue = 0
        max_index = 0
        for curr_index in range(start_index, end_index):
            temp = range_fft[frame_index, curr_index]
            if abs(temp) > maxvalue:
                maxvalue = abs(temp)
                max_index = curr_index
        temp = range_fft[frame_index, max_index]
        result[frame_index] = range_fft[frame_index, max_index]
        phase[frame_index] = math.atan2(temp.imag, temp.real)

    phase_unwrap = np.unwrap(phase)

    # diff_phase 부분
    phase_unwrap = np.concatenate(([0], phase_unwrap))
    len_ = len(phase_unwrap)
    diff_phase = np.zeros(len_ - 1)
    for i in range(1, len_):
        diff_phase[i - 1] = phase_unwrap[i] - phase_unwrap[i - 1]

    # np.convolve 부분
    phase_remove = np.convolve(diff_phase, np.ones(3) / 3, 'same')

    # 호흡수 부분
    fs = 20
    f1_breath = (8/60) / (fs/2)  #분단 8회로 가정
    f2_breath = (25/60) / (fs/2)  #분당 25회로 가정 
    sos_breath = butter(4, [f1_breath, f2_breath], btype='bandpass', output='sos')
    res_breath = sosfilt(sos_breath, phase_remove)

    # 심박수 부분
    f1_heart = (40/60) / (fs/2)  #분단 70회로 가정
    f2_heart = (110/60) / (fs/2)  #분당 150회로 가정 
    sos_heart = butter(8, [f1_heart, f2_heart], btype='bandpass', output='sos')
    res_heart = sosfilt(sos_heart, phase_remove)

    # FFT 진행 부분
    fft_breath = np.abs(np.fft.fft(res_breath)) ** 2
    fft_heart = np.abs(np.fft.fft(res_heart)) ** 2

    # 호흡수 피크 추출
    breath_start_freq_index = math.floor((8  / 60) / (20 / numchirps)) # 8회 스타트
    breath_end_freq_index = math.ceil((25  / 60) / (20 / numchirps)) #  20회 끝
    breath_peak_values = np.zeros(int(numchirps/2)) 
    breath_peak_index = np.zeros(int(numchirps/2))
    breath_max_num_peaks_spectrum = 4
    breath_num_peaks = 0

    for i in range(breath_start_freq_index, breath_end_freq_index):
        if fft_breath[i] > fft_breath[i - 1] and fft_breath[i] > fft_breath[i + 1]:
            breath_peak_index[breath_num_peaks] = i
            breath_peak_values[breath_num_peaks] = fft_breath[i]
            breath_num_peaks += 1
    if breath_num_peaks < breath_max_num_peaks_spectrum:
        index_num_peaks = breath_num_peaks
    else:
        index_num_peaks = breath_max_num_peaks_spectrum

    breath_peak_index_sorted = np.zeros(index_num_peaks)
    if index_num_peaks != 0:
        for i in range(index_num_peaks):
            idx = np.argmax(breath_peak_values)
            breath_peak_index_sorted[i] = idx
            breath_peak_values[idx] = 0
        max_index_breath_spect = breath_peak_index[int(breath_peak_index_sorted[0])]
    else:
        max_index_breath_spect = np.argmax(fft_breath[breath_start_freq_index:breath_end_freq_index])

    breath = 60 * (max_index_breath_spect-1) * (20 / numchirps)     

    # 심박수 피크 추출
    heart_start_freq_index = math.floor((40  / 60) / (20 / numchirps)) # 70회 스타트
    heart_end_freq_index = math.ceil((110  / 60) / (20 / numchirps)) # 150회 끝
    heart_peak_values = np.zeros(int(numchirps/2)) 
    heart_peak_index = np.zeros(int(numchirps/2))   
    heart_max_num_peaks_spectrum = 4
    heart_num_peaks = 0

    for i in range(heart_start_freq_index, heart_end_freq_index):
        if fft_heart[i] > fft_heart[i - 1] and fft_heart[i] > fft_heart[i + 1]:
            heart_peak_index[heart_num_peaks] = i
            heart_peak_values[heart_num_peaks] = fft_heart[i]
            heart_num_peaks += 1
    if heart_num_peaks < heart_max_num_peaks_spectrum:
        index_num_peaks = heart_num_peaks
    else:
        index_num_peaks = heart_max_num_peaks_spectrum

    heart_peak_index_sorted = np.zeros(index_num_peaks)
    if index_num_peaks != 0:
        for i in range(index_num_peaks):
            idx = np.argmax(heart_peak_values)
            heart_peak_index_sorted[i] = idx
            heart_peak_values[idx] = 0
        max_index_heart_spect = heart_peak_index[int(heart_peak_index_sorted[0])]
    else:
        max_index_heart_spect = np.argmax(fft_heart[heart_start_freq_index:heart_end_freq_index])

    heart = 60 * (max_index_heart_spect-1) * (20 / numchirps)

    return breath, heart