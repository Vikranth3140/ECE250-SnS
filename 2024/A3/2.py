import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Fourier coefficients
def compute_fourier_coefficients(T1, T, k_values):
    c_k = np.zeros_like(k_values, dtype=np.float64)
    for i, k in enumerate(k_values):
        if k == 0:
            c_k[i] = 2 * T1 / T  # DC component
        else:
            w_k = 2 * np.pi * k / T
            c_k[i] = (T * np.sin(w_k * T1)) / (np.pi * k)
    return c_k

# Parameters
T1 = 1  # Given T1 = 2 seconds (normalized to 1 for simplicity)
k_values = np.arange(-50, 51)  # Fourier coefficients for k = -50 to 50

# Case a: T = 4T1
T_a = 4 * T1
c_k_a = compute_fourier_coefficients(T1, T_a, k_values)

# Case b: T = 8T1
T_b = 8 * T1
c_k_b = compute_fourier_coefficients(T1, T_b, k_values)

# Case c: T = 16T1
T_c = 16 * T1
c_k_c = compute_fourier_coefficients(T1, T_c, k_values)

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(k_values, c_k_a, basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 4T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.subplot(3, 1, 2)
plt.stem(k_values, c_k_b, basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 8T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.subplot(3, 1, 3)
plt.stem(k_values, c_k_c, basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 16T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.tight_layout()
plt.show()
