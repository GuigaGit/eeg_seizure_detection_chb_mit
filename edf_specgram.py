import mne
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the EDF file
file_path = 'chb21_21.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Optional: Apply a notch filter to remove power line noise (e.g., 60 Hz)
# raw.notch_filter(freqs=60.0)

# 2. Select a channel to analyze
for idx, ch in enumerate(raw.ch_names):
    print(f"{idx}: {ch}")
    ch_name = raw.ch_names[idx] 

    # Extract the data array and sampling frequency for the selected channel
    # get_data() returns a 2D array of shape (n_channels, n_times)
    data, times = raw.get_data(picks=ch_name, return_times=True)
    channel_data = data[0]
    sfreq = raw.info['sfreq']

    # 3. Compute and plot the Spectrogram
    plt.figure(figsize=(12, 6))

    # plt.specgram calculates the STFT and plots the result
    # NFFT: The number of data points used in each block for the FFT (defines frequency resolution)
    # noverlap: The number of points of overlap between blocks (defines time resolution)

    channel_data = channel_data / np.max(np.abs(channel_data))  # Normalize for better visualization
    plt.specgram(
        channel_data, 
        Fs=sfreq, 
        NFFT=int(sfreq * 2),      # 4-second windows (adjust as needed)
        noverlap=int(sfreq * 1),  # 0.5-second overlap
        cmap='viridis'
    )

    # 4. Format the plot
    plt.title(f'Spectrogram over Time: Channel {file_path}-{ch_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.ylim(0, 128) 

    plt.colorbar(label='Power/Intensity (dB)')
    plt.tight_layout()
    plt.show()