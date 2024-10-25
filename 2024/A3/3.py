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

# Explanation of the results:

# 1. Prominent Frequencies:
# The magnitude spectrum shows two prominent peaks:
# - One at 5 Hz
# - Another at 12 Hz
# These correspond to the two cosines present in the original signal x[n].

# 2. Amplitude of the Peaks:
# The peak at 5 Hz has an amplitude of about 32, and the peak at 12 Hz has an amplitude of about 16.
# This is expected because:
# - The 5 Hz cosine has an amplitude of 1, so the FFT gives a peak of 32 (related to the number of samples N=64).
# - The 12 Hz cosine has an amplitude of 0.5, so the FFT gives a peak of about 16 (half the amplitude of the 5 Hz cosine).

# 3. Symmetry in the Full Spectrum:
# In the full FFT spectrum, you see peaks at both positive and negative frequencies.
# The signal is real-valued, so the FFT is symmetric. You see mirror image peaks at +5 Hz, -5 Hz, +12 Hz, and -12 Hz.

# 4. Positive Frequency Spectrum:
# The second plot focuses on the positive frequencies. It shows two peaks at 5 Hz and 12 Hz.
# These peaks represent the two cosine waves in the original signal, confirming the frequency components.