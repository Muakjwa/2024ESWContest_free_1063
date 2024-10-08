import numpy as np
import math


def peakcure(data):
    """
    Extracts the peak value and phase information of a target range bin from a radar data matrix.

    Args:
        data (ndarray): The radar data matrix, where each row represents a range-bin and each column represents a frame.

    Returns:
        tuple: A tuple containing three ndarrays:
               - Peakcurve: The extracted peak value for each frame.
               - phase: The corresponding phase angle of the peak value.
               - phase_unwrap: The unwrapped phase angle that has been corrected for any periodicity caused by phase wrapping.
    """
    start_index = 12
    end_index = 66
    # end_index = 120
    num_frame = data.shape[0]
    # num_sample = data.shape[1]
    result = np.zeros(num_frame, dtype=complex)
    phase = np.zeros(num_frame)

    for frame_index in range(num_frame):
        maxvalue = 0
        max_index = 0
        for curr_index in range(start_index, end_index):
            temp = data[frame_index, curr_index]
            if abs(temp) > maxvalue:
                maxvalue = abs(temp)
                max_index = curr_index

        temp = data[frame_index, max_index]
        result[frame_index] = data[frame_index, max_index]
        phase[frame_index] = math.atan2(temp.imag, temp.real)

    phase_unwrap = np.unwrap(phase)
    return result, phase, phase_unwrap
