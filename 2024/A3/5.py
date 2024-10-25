import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

plot_dir = '5_Plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fs = 1000
t = np.arange(0, 2, 1/fs)

x = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 100 * t)

h1 = np.array([0.1, 0.15, 0.5, 0.15, 0.1])

b, a = butter(2, 50 / (fs / 2), btype='low')

yfir = np.convolve(x, h1, mode='same')

yiir = filtfilt(b, a, x)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.title('Original Signal x[n]')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, yfir, label='FIR Filtered Signal', color='green')
plt.title('FIR Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, yiir, label='IIR Filtered Signal', color='red')
plt.title('IIR Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(plot_dir, 'time_domain_signals.png'))
plt.show()

def plot_fft(signal, fs, title, filename):
    N = len(signal)
    f = np.fft.fftfreq(N, 1/fs)
    fft_vals = np.fft.fft(signal)
    plt.plot(f[:N // 2], np.abs(fft_vals[:N // 2]))
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plot_fft(x, fs, 'FFT of Original Signal', 'fft_original_signal.png')

plt.subplot(3, 1, 2)
plot_fft(yfir, fs, 'FFT of FIR Filtered Signal', 'fft_fir_filtered_signal.png')

plt.subplot(3, 1, 3)
plot_fft(yiir, fs, 'FFT of IIR Filtered Signal', 'fft_iir_filtered_signal.png')

plt.figure(figsize=(10, 6))
w, h = freqz(h1, worN=8000)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), label="FIR Filter Response", color='green')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fir_filter_response.png'))
plt.show()

plt.figure(figsize=(10, 6))
w, h = freqz(b, a, worN=8000)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), label="IIR Filter Response", color='red')
plt.title('IIR Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'iir_filter_response.png'))
plt.show()