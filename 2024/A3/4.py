import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 500)

x_t = np.exp(-t**2 / (2 * 0.1**2))

X_fft = np.fft.fft(x_t)
X_magnitude = np.abs(X_fft)

frequencies = np.fft.fftfreq(t.size, t[1] - t[0])

plt.figure(figsize=(10, 6))
plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(X_magnitude))
plt.title('Magnitude Spectrum of the Gaussian Pulse')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()