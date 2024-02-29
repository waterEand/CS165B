import matplotlib.pyplot as plt
import numpy as np

# Given points
x = np.array([-2, 1, 3])
y = np.array([-1, 1, 2])

# Calculate the coefficients of the least squares regression line
m = 0.605
b = 0.263

# Plot the points
plt.scatter(x, y, color='blue')

# Plot the regression line
x_line = np.linspace(-3, 4, 400)
y_line = m*x_line + b
plt.plot(x_line, y_line, color='red')

# Set labels
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()