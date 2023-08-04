import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

image = Image.open('image-data/1/13.png').convert('L')
image = np.asarray(image)
mat = image.copy()
xx, yy = np.mgrid[0:mat.shape[0], 0:mat.shape[1]]
print(xx.shape)
print(yy.shape)
print(mat.shape)

print(np.where(mat==np.max(mat)))
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(projection='3d')
ax.plot_surface(yy, xx, mat, rstride=1, cstride=1,
                linewidth=1)
ax.view_init(elev=0, azim=90)
ax.grid()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


