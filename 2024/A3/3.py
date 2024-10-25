import numpy as np
import matplotlib.pyplot as plt

n = np.arange(64)

x = np.cos(2 * np.pi * 5 * n / 64) + 0.5 * np.cos(2 * np.pi * 12 * n / 64)

X_fft = np.fft.fft(x)
X_magnitude = np.abs(X_fft)

frequencies = np.fft.fftfreq(64) * 64

plt.figure(figsize=(10, 6))
plt.stem(frequencies, X_magnitude, basefmt=" ", use_line_collection=True)
plt.title('Magnitude Spectrum of x[n]')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.stem(frequencies[:64//2], X_magnitude[:64//2], basefmt=" ", use_line_collection=True)
plt.title('Magnitude Spectrum (Positive Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()