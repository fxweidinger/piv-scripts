import numpy as np
import matplotlib.pyplot as plt


def calculate_flow_profile(image1, image2):
    # Calculate the cross-correlation using Fourier Transform
    corr = np.fft.fftshift(
        np.fft.ifft2(np.conj(np.fft.fft2(image1)) * np.fft.fft2(image2)))

    # Calculate the flow profile along the x-axis
    flow_profile = np.mean(corr, axis=0)

    return flow_profile


def generate_flow_pattern(width, height, num_particles, max_velocity):
    x = np.random.randint(0, width, num_particles)
    y = np.random.randint(0, height, num_particles)

    # Generate velocities with higher magnitudes near the center of the duct
    center_x = width // 2
    center_y = height // 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    velocity_scale = max_velocity * (1 - distance / (width / 2))

    u = np.random.uniform(-velocity_scale, velocity_scale, num_particles)
    v = np.random.uniform(-velocity_scale, velocity_scale, num_particles)

    return x, y, u, v


def apply_boundary_conditions(x, y, u, v, width, height):
    x[x < 0] = 0
    y[y < 0] = 0
    x[x >= width] = width - 1
    y[y >= height] = height - 1
    return x, y, u, v


def generate_duct_flow_image(width, height, num_particles, max_velocity):
    image1 = np.zeros((height, width))
    image2 = np.zeros((height, width))
    x, y, u, v = generate_flow_pattern(width, height, num_particles, max_velocity)
    x, y, u, v = apply_boundary_conditions(x, y, u, v, width, height)

    for i in range(num_particles):
        image1[y[i], x[i]] = 1.0
        if (y[i] + int(v[i])) > width-1 or (x[i] + int(u[i])) > width-1:
            break
        else:
            image2[y[i] + int(v[i]), x[i] + int(u[i])] = 1.0

    return image1, image2


# Example usage
width = 500
height = 500
num_particles = 500
max_velocity = 10.3

image1, image2 = generate_duct_flow_image(width, height, num_particles, max_velocity)
flow_profile = calculate_flow_profile(image1, image2)

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')

plt.subplot(2, 1, 1)
x = np.arange(len(flow_profile))
plt.xticks([0, 50, 100, 150, 200, 250, 300])
plt.plot(x, flow_profile)

plt.show()
