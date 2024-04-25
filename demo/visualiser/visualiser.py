from matplotlib.image import NEAREST
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize

filename = sys.argv[1]

with open(filename, 'r') as file:
    first_line = file.readline().split()
    x_min, x_max = float(first_line[0]), float(first_line[1])
    y_min, y_max = float(first_line[2]), float(first_line[3])
    width, height = int(first_line[4]), int(first_line[5])
    data = np.array([list(map(int, line.split())) for line in file], dtype=int)


log_data = np.log(np.log2(data + 1))

norm = Normalize(vmin=np.min(data), vmax=np.max(data))

plt.figure(figsize=(10, 10))
plt.imshow(norm(log_data), cmap='magma_r', extent=(x_min, x_max, y_min, y_max), interpolation='nearest')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.show()
