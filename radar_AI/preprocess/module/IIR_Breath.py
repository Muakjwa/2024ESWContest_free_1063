import numpy as np
from scipy.signal import butter, sosfilt


def iir_breath(n: int, phase: np.ndarray):
    fs = 20 
    f1 = 0.1 / (fs/2)
    f2 = 0.6 / (fs/2)
    
    sos = butter(n, [f1, f2], btype='bandpass', output='sos')
    res = sosfilt(sos, phase)
    return res



