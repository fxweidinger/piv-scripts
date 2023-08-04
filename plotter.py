import math

import numpy as np
import matplotlib.pyplot as plt

flow= 50*(1/60)*1000000*(1/math.pi)*(1/(24.5*24.5))*(1/1000)
print(flow)
diplacements = np.array(
    [0.92268, 3.1928, 3.28434, 3.588, 3.7949, 3.90, 3.89, 3.79, 3.66, 3.45,
     3.2])
position = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160])

plt.rcParams["figure.figsize"]=(10,10)
plt.subplots_adjust(hspace=None)
plt.subplot(2,1,1)
plt.ylabel("Pixel Displacement")
plt.xlabel("Radial Position [px]")
plt.plot(position, diplacements)
plt.grid()
velocity = (diplacements*100e-6)/800e-6


plt.subplot(2,1,2)
plt.title(f'Mean Velocity = {np.mean(velocity)}')

plt.ylabel("m/s")
plt.xlabel("Radial Position [px]")
plt.plot(position,velocity)
plt.grid()
plt.show()
print()