import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, lfilter

# Ensure the directory '5_Plots/' exists
plot_dir = '5_Plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Step 1: Generate the composite signal x[n]
fs = 1000  # Sampling frequency (1 / sampling interval)
t = np.arange(0, 2, 1/fs)  # Time vector from 0 to 2 seconds, with a sampling interval of 0.001 seconds

# Composite signal: x[n] = cos(2 * pi * 10 * n) + 0.5 * cos(2 * pi * 100 * n)
x = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 100 * t)

# Step 2: FIR filter definition
h1 = np.array([0.1, 0.15, 0.5, 0.15, 0.1])  # FIR filter impulse response

# Step 3: IIR filter design (Butterworth filter with fc = 50 Hz)
fc = 50  # Cutoff frequency
b, a = butter(2, fc / (fs / 2), btype='low')  # 2nd-order Butterworth filter

# Step 4: Apply the FIR filter to the signal
yfir = np.convolve(x, h1, mode='same')  # FIR filter application

# Step 5: Apply the IIR filter to the signal
yiir = filtfilt(b, a, x)  # IIR filter application using zero-phase filtering

# Step 6: Plot the original signal and the filtered signals
plt.figure(figsize=(12, 8))

# Plot original signal
plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.title('Original Signal x[n]')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Plot FIR filtered signal
plt.subplot(3, 1, 2)
plt.plot(t, yfir, label='FIR Filtered Signal', color='green')
plt.title('FIR Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Plot IIR filtered signal
plt.subplot(3, 1, 3)
plt.plot(t, yiir, label='IIR Filtered Signal', color='red')
plt.title('IIR Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()

# Save the plot as a file in the '5_Plots/' directory
plt.savefig(os.path.join(plot_dir, 'time_domain_signals.png'))
plt.show()

# Step 7: Frequency domain analysis - FFT
def plot_fft(signal, fs, title, filename):
    N = len(signal)
    f = np.fft.fftfreq(N, 1/fs)  # Frequency vector
    fft_vals = np.fft.fft(signal)  # FFT of the signal
    plt.plot(f[:N // 2], np.abs(fft_vals[:N // 2]))  # Plot only positive frequencies
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))  # Save the plot in '5_Plots/'
    plt.show()

# Plot FFT of original, FIR filtered, and IIR filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plot_fft(x, fs, 'FFT of Original Signal', 'fft_original_signal.png')

plt.subplot(3, 1, 2)
plot_fft(yfir, fs, 'FFT of FIR Filtered Signal', 'fft_fir_filtered_signal.png')

plt.subplot(3, 1, 3)
plot_fft(yiir, fs, 'FFT of IIR Filtered Signal', 'fft_iir_filtered_signal.png')

# Frequency response of FIR filter
plt.figure(figsize=(10, 6))
w, h = freqz(h1, worN=8000)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), label="FIR Filter Response", color='green')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'fir_filter_response.png'))  # Save FIR response plot
plt.show()

# Frequency response of IIR filter
plt.figure(figsize=(10, 6))
w, h = freqz(b, a, worN=8000)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), label="IIR Filter Response", color='red')
plt.title('IIR Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'iir_filter_response.png'))  # Save IIR response plot
plt.show()
