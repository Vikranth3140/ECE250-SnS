import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
f1 = 5  # Frequency of the first cosine wave (Hz)
f2 = 12  # Frequency of the second cosine wave (Hz)
N = 64  # Number of samples
n = np.arange(N)  # Sample indices (n = 0, 1, ..., N-1)

# Step 2: Generate the signal x[n]
x = np.cos(2 * np.pi * f1 * n / N) + 0.5 * np.cos(2 * np.pi * f2 * n / N)

# Step 3: Compute the FFT of x[n]
X_fft = np.fft.fft(x)  # FFT of the signal
X_magnitude = np.abs(X_fft)  # Magnitude of the FFT

# Step 4: Frequency axis for plotting (0 to N-1)
frequencies = np.fft.fftfreq(N) * N  # Frequency axis normalized to [0, N-1]

# Plot the magnitude spectrum
plt.figure(figsize=(10, 6))
plt.stem(frequencies, X_magnitude, basefmt=" ", use_line_collection=True)
plt.title('Magnitude Spectrum of x[n]')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# To see just the positive frequencies:
plt.figure(figsize=(10, 6))
plt.stem(frequencies[:N//2], X_magnitude[:N//2], basefmt=" ", use_line_collection=True)
plt.title('Magnitude Spectrum (Positive Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
