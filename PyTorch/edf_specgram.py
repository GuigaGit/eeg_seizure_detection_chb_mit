"""
Plots a spectrogram (power at each frequency, over time) for every channel
of one .edf recording, one channel at a time. This script has no functions -
it's a flat, top-to-bottom script meant to be read/tweaked directly - so
here's what each numbered step below does:

  1. Load the .edf file with MNE (see `DATASET_DIR` for where it looks).
  2. Loop over every channel in the recording.
  3. For the current channel: normalize its amplitude to [-1, 1] (so the dB
     scale below is comparable across channels regardless of their raw
     voltage scale), then compute its spectrogram via `torch.stft` - a
     Short-Time Fourier Transform, i.e. many small windowed FFTs computed
     over overlapping time slices, whose squared magnitude (converted to
     decibels here) gives power-per-frequency-per-time-slice. This replaces
     matplotlib's `plt.specgram`, which does the same computation
     internally but wasn't using torch.
  4. Plot the resulting time-frequency power map with `plt.pcolormesh`,
     capped to 0-128 Hz (covers all clinically relevant EEG bands).

Each channel opens its own plot window (`plt.show()` blocks until closed),
so this is meant for interactive/manual inspection, not batch processing.
"""
import mne
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pasta com os arquivos .edf do CHB-MIT. Cada máquina pode ter o dataset em
# um lugar diferente, então isso é configurável via variável de ambiente:
#   export CHB_MIT_DATASET_DIR=/caminho/para/dataset_chbmit
# Se não for definida, usa a pasta de exemplo dentro do próprio repositório.
DATASET_DIR = os.environ.get('CHB_MIT_DATASET_DIR', os.path.join(REPO_ROOT, 'dataset_chbmit'))

# 1. Load the EDF file
file_path = os.path.join(DATASET_DIR, 'chb21', 'chb21_21.edf')
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
