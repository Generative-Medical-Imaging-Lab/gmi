import torch
import numpy as np

# Test the polar resampler
from gmi.linear_operator import PolarCoordinateResampler

# Create test data
num_row, num_col = 64, 64
theta_values = torch.linspace(0, 2 * np.pi, 32)
radius_values = torch.linspace(0, 30, 16)

# Create a simple test image (a circle)
x = torch.zeros(1, 1, num_row, num_col)
center_row, center_col = num_row // 2, num_col // 2
for i in range(num_row):
    for j in range(num_col):
        dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
        if dist < 20:
            x[0, 0, i, j] = 1.0

print("Input image shape:", x.shape)
print("Theta values shape:", theta_values.shape)
print("Radius values shape:", radius_values.shape)

# Create polar resampler
polar_resampler = PolarCoordinateResampler(
    num_row=num_row,
    num_col=num_col,
    theta_values=theta_values,
    radius_values=radius_values,
    interpolator='lanczos'
)

# Apply forward transformation
result = polar_resampler.forward(x)
print("Polar result shape:", result.shape)

# Apply transpose transformation
back_result = polar_resampler.transpose(result)
print("Back-transformed result shape:", back_result.shape)

# Check that the result makes sense
print("Max value in polar result:", torch.max(result))
print("Min value in polar result:", torch.min(result))
print("Max value in back-transformed result:", torch.max(back_result))
print("Min value in back-transformed result:", torch.min(back_result))

print("Polar resampler test completed successfully!") 