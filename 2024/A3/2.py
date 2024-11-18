import numpy as np
import matplotlib.pyplot as plt
import math

def fourier_coeff(T, k_values):
    c_k = np.zeros_like(k_values, dtype=np.float64)
    for i, k in enumerate(k_values):
        if k == 0:
            c_k[i] = 2 * 1 / T
        else:
            w_k = 2 * np.pi * k / T
            c_k[i] = (T * np.sin(w_k * 1)) / (np.pi * k)
    return c_k

k_values = np.arange(-50, 51)

# a
T_a = 4 * 1
c_k_a = fourier_coeff(T_a, k_values)

# b
T_b = 8 * 1
c_k_b = fourier_coeff(T_b, k_values)

# c
T_c = 16 * 1
c_k_c = fourier_coeff(T_c, k_values)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(k_values, abs(c_k_a), basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 4T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.subplot(3, 1, 2)
plt.stem(k_values, abs(c_k_b), basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 8T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.subplot(3, 1, 3)
plt.stem(k_values, abs(c_k_c), basefmt=" ", use_line_collection=True)
plt.title("Fourier Coefficients for T = 16T1")
plt.xlabel("k")
plt.ylabel("$c_k$")

plt.tight_layout()
plt.show()