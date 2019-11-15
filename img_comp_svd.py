import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.linalg import matrix_rank
from PIL import Image

# Read the image and convert into black and white
img = Image.open(sys.argv[1])
imggray = img.convert('LA')

# Let's now convert the image data into a numpy matrix
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray')
plt.title("Original gray scale image with %s rank and %s bytes" % (matrix_rank(imgmat), imgmat.size))
plt.show();

# Compute the Singular Value Decomposition
U, sigma, V = np.linalg.svd(imgmat)

# Reconstruction of the image with the user input rank
rank = int(sys.argv[2])
reConImg = np.matrix(U[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(V[:rank, :])
plt.imshow(reConImg, cmap='gray')
plt.title("Reconstructed image with wth %s rank and %s bytes" % (rank, (rank*(reConImg.shape[0] + reConImg.shape[1] + 1))))
plt.show()
