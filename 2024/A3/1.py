import numpy as np
import matplotlib.pyplot as plt

n_x = np.arange(-10, 11)
n_h = np.arange(0, 50)

x = np.ones_like(n_x)
h = np.exp(-0.1 * n_h)

y_manual = np.zeros(len(n_x) + len(n_h) - 1)

for i in range(len(y_manual)):
    for j in range(len(n_x)):
        if 0 <= i - j < len(h):
            y_manual[i] += x[j] * h[i - j]

y_np_convolve = np.convolve(x, h)

n_y_manual = np.arange(-10, len(y_manual) - 10)
n_y_np = np.arange(-10, len(y_np_convolve) - 10)

plt.figure(figsize=(10, 6))
plt.stem(n_y_manual, y_manual, linefmt='blue', markerfmt='bo', basefmt=" ", label='Manual Convolution')
plt.stem(n_y_np, y_np_convolve, linefmt='red', markerfmt='ro', basefmt=" ", label='np.convolve() Convolution')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Comparison of Manual Convolution and np.convolve() Convolution')
plt.legend()
plt.grid(True)
plt.show()