import numpy as np
import matplotlib.pyplot as plt

def plot_range_fft(data, radar):
    BandWidth = radar.config.end_frequency_Hz - radar.config.start_frequency_Hz
    delta_r = radar.constant.c / (2 * BandWidth)
    r_max = delta_r / 2 * radar.config.num_samples_per_chirp

    plt.figure(figsize = (12, 4))
    plt.imshow(abs(data), aspect='auto', extent=[0, r_max, data.shape[0], 0], cmap='viridis')
    plt.colorbar()
    plt.title('Range FFT')
    plt.xlabel('Range')
    plt.ylabel('Frame')
    plt.show()


def plot_bio_rythm(data, radar, start, end):
    BandWidth = radar.config.end_frequency_Hz - radar.config.start_frequency_Hz
    delta_r = radar.constant.c / (2 * BandWidth)
    r_max = delta_r / 2 * radar.config.num_samples_per_chirp

    plt.figure(figsize = (20, 8))
    plt.imshow(np.transpose(abs(data))[start:end, :], aspect='auto', extent=[0, data.shape[0], r_max * 100 / 256, r_max * 20 / 256], cmap='viridis')
    plt.colorbar()
    plt.title('Bio Rythm')
    plt.ylabel('Range')
    plt.xlabel('Frame')
    plt.show()


def plot_RD_Map(data, radar):
    BandWidth = radar.config.end_frequency_Hz - radar.config.start_frequency_Hz
    delta_r = radar.constant.c / (2 * BandWidth)
    r_max = delta_r / 2 * radar.config.num_samples_per_chirp

    v_max = 0.005 / (4 * radar.config.chirp_repetition_time_s)

    plt.figure(figsize = (12, 4))
    plt.imshow(abs(data), aspect='auto', extent=[0, r_max, 0, v_max], cmap='viridis')
    plt.colorbar()
    plt.title('RD-Map')
    plt.xlabel('Range')
    plt.ylabel('Velocity')
    plt.show()


def plot_VMD(before, after):
    plt.figure(figsize = (15, 9))

    # Plot original signal (phase_unwrap)
    plt.subplot(3, 1, 1)
    plt.plot(before)
    plt.title("Original signal")
    plt.xlabel("Frame")

    # Plot VMD changed signal (1D)
    plt.subplot(3, 1, 2)
    plt.plot(after.T)
    plt.title("Decomposed modes")
    plt.xlabel("Frame")
    plt.legend(["Mode %d" % m_i for m_i in range(after.shape[0])])
    
    # Plot VMD changed (2D)
    plt.subplot(3, 1, 3)
    plt.imshow(after, aspect='auto', extent=[0, after.shape[1], after.shape[0], 0], cmap='viridis')
    plt.xlabel("Frame")
    plt.tight_layout()


def plot_compare_output(predict, gt, y_start = 25, y_end = 100):
    plt.figure(figsize = (15, 7))

    # Plot Predicted Output
    plt.subplot(2, 1, 1)
    plt.plot(predict)
    plt.title("Predict")
    plt.ylim(y_start, y_end)

    # Plot GroundTruth
    plt.subplot(2, 1, 2)
    plt.plot(gt)
    plt.title("Ground Truth")
    plt.ylim(y_start, y_end)

    plt.tight_layout()