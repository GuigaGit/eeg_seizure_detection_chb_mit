import mne
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the EDF file
file_path = os.path.join(REPO_ROOT, 'dataset_chbmit', 'chb21', 'chb21_21.edf')
raw = mne.io.read_raw_edf(file_path, preload=True)

# Optional: Apply a notch filter to remove power line noise (e.g., 60 Hz)
# raw.notch_filter(freqs=60.0)

# 2. Select a channel to analyze
for idx, ch in enumerate(raw.ch_names):
    print(f"{idx}: {ch}")
    ch_name = raw.ch_names[idx]

    # Extract the data array and sampling frequency for the selected channel
    data, times = raw.get_data(picks=ch_name, return_times=True)
    channel_data = data[0]
    sfreq = raw.info['sfreq']

    channel_data = channel_data / np.max(np.abs(channel_data))  # Normalize for better visualization
    signal = torch.as_tensor(channel_data, dtype=torch.float32)

    # 3. Compute the Spectrogram via torch.stft (replaces plt.specgram)
    n_fft = int(sfreq * 2)       # 2-second windows (adjust as needed)
    noverlap = int(sfreq * 1)    # 1-second overlap
    hop_length = n_fft - noverlap
    window = torch.hann_window(n_fft)

    stft = torch.stft(
        signal, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, center=True, return_complex=True
    )
    power_db = 20 * torch.log10(stft.abs() + 1e-10)

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq).numpy()
    frame_times = (torch.arange(power_db.shape[1]) * hop_length / sfreq).numpy()

    # 4. Plot the Spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(frame_times, freqs, power_db.numpy(), cmap='viridis', shading='auto')

    plt.title(f'Spectrogram over Time: Channel {file_path}-{ch_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.ylim(0, 128)

    plt.colorbar(label='Power/Intensity (dB)')
    plt.tight_layout()
    plt.show()
