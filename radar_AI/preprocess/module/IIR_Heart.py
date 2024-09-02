import numpy as np
from scipy.signal import butter, sosfilt


def iir_heart(n: int, phase: np.ndarray):
    fs = 20  
    f1 = 0.8 / (fs/2) 
    f2 = 2 / (fs/2) 
    
    sos = butter(n, [f1, f2], btype='bandpass', output='sos')
    res = sosfilt(sos, phase)
    return res



