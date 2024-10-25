import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters for the Gaussian pulse
sigma = 0.1  # Controls the width of the Gaussian pulse
t = np.linspace(-1, 1, 500)  # Time vector from -1 to 1 with 500 samples

# Generate the Gaussian pulse x(t)
x_t = np.exp(-t**2 / (2 * sigma**2))

# Step 2: Compute the FFT of the Gaussian pulse
X_fft = np.fft.fft(x_t)  # Perform the FFT of the signal
X_magnitude = np.abs(X_fft)  # Compute the magnitude of the FFT

# Normalize frequencies (FFT returns bins, we map them to real frequencies)
frequencies = np.fft.fftfreq(t.size, t[1] - t[0])  # Frequency axis

# Step 3: Plot the magnitude spectrum of the FFT
plt.figure(figsize=(10, 6))
plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(X_magnitude))  # FFT shifted to center the spectrum
plt.title('Magnitude Spectrum of the Gaussian Pulse')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Explanation of the result:

# 1. The Gaussian pulse x(t) is symmetric in time (centered at t=0), and its FFT magnitude spectrum
#    will also be symmetric. The result is a smooth, bell-shaped curve in the frequency domain.

# 2. The width of the magnitude spectrum depends on sigma. A smaller sigma results in a wider pulse
#    in the time domain, which in turn leads to a narrower magnitude spectrum in the frequency domain.

# 3. The frequency domain interpretation of a Gaussian pulse is that it has a concentrated energy around
#    low frequencies, with rapidly decreasing amplitude as the frequency increases.
#    This reflects the fact that the Gaussian pulse is a smooth, low-frequency signal in the time domain.