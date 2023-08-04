import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def generate_flow_pattern(width, height, num_particles, max_velocity):
    x = np.random.randint(0, width, num_particles)
    y = np.random.randint(0, height, num_particles)
    u = np.random.uniform(-max_velocity, max_velocity, num_particles)
    v = np.random.uniform(-max_velocity, max_velocity, num_particles)
    return x, y, u, v

def apply_boundary_conditions(x, y, u, v, width, height):
    x[x < 0] = 0
    y[y < 0] = 0
    x[x >= width] = width - 1
    y[y >= height] = height - 1
    return x, y, u, v

def generate_double_exposure_image(width, height, num_particles, max_velocity):
    image1 = np.zeros((height, width))
    image2 = np.zeros((height, width))
    x, y, u, v = generate_flow_pattern(width, height, num_particles, max_velocity)
    x, y, u, v = apply_boundary_conditions(x, y, u, v, width, height)

    for i in range(num_particles):
        image1[y[i], x[i]] = 1.0
        image2[y[i] + int(v[i]), x[i] + int(u[i])] = 1.0

    return image1, image2

# Example usage
width = 128
height = 128
num_particles = 100
max_velocity = 5.0

image1, image2 = generate_double_exposure_image(width, height, num_particles, max_velocity)
image3 = (image1+image2)/2
plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')

plt.subplot(1, 3, 3)
plt.imshow(image3, cmap='gray')
plt.title('Image 2')
plt.show()
io.imsave('image-data/output.tiff', image3)