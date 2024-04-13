import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

with open(filename, 'r') as file:
    first_line = file.readline().split()
    x_min, x_max = float(first_line[0]), float(first_line[1])
    y_min, y_max = float(first_line[2]), float(first_line[3])
    width, height = int(first_line[4]), int(first_line[5])
    data = np.array([list(map(int, line.split())) for line in file], dtype=int)
plt.figure(figsize=(10, 10))
plt.imshow(data, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar()
plt.title('Mandelbrot Set')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.show()
