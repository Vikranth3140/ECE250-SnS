import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the rectangular pulse x[n] and the exponential decay h[n]

# Range of n for x[n] and h[n]
n_x = np.arange(-10, 11)  # x[n] from -10 to 10
n_h = np.arange(0, 50)    # h[n] for n >= 0

# Define x[n] (rectangular pulse)
x = np.ones_like(n_x)

# Define h[n] (exponentially decaying signal)
h = np.exp(-0.1 * n_h)

# Step 2: Perform manual convolution
# To perform convolution, we pad x and h appropriately
y_manual = np.zeros(len(n_x) + len(n_h) - 1)

for i in range(len(y_manual)):
    for k in range(len(n_x)):
        if 0 <= i - k < len(h):  # Ensure valid index for h[n-k]
            y_manual[i] += x[k] * h[i - k]

# Step 3: Compare with np.convolve()
y_np_convolve = np.convolve(x, h)

# Step 4: Plot both results for comparison
n_y_manual = np.arange(-10, len(y_manual) - 10)  # n range for manual convolution
n_y_np = np.arange(-10, len(y_np_convolve) - 10) # n range for np.convolve

plt.figure(figsize=(10, 6))
plt.plot(n_y_manual, y_manual, label='Manual Convolution', color='blue', linewidth=2)
plt.plot(n_y_np, y_np_convolve, label='np.convolve()', color='red', linewidth=1)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.title('Comparison of Manual Convolution and np.convolve()')
plt.legend()
plt.grid(True)
plt.show()
