# This import registers the 3D projection, but is otherwise unused.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans

# Image path
imgs_base = "data/Images/Original Images/"
n_imgs = 20
imgs = [cv.imread(imgs_base + "img_{0:03d}.jpg".format(i)) for i in range(1, n_imgs + 1)]


def normalize(seq, input_start, input_end, output_start, output_end):
    factor = (output_end - output_start) / (input_end - input_start)
    return (output_start + factor * (seq - input_start)).astype(np.uint8)


n_clusters = 3

while n_clusters < 5:
    for img in imgs:
        # Prepare data for k-Means algorithm
        cvectors = img.reshape(-1, 3)

        # Create the model
        kmeans = KMeans(n_clusters=n_clusters)
        # Fit the model
        model = kmeans.fit(cvectors)
        # Get cluster id for each data point in data set
        pred = model.predict(cvectors)

        # Reshape it as original image
        r_pred = pred.reshape(img.shape[:-1])
        # Convert dtype to np.uint8
        r_pred = r_pred.astype(np.uint8)
        # To see difference among clusters
        # Map cluster ids from range(0, n_clusters-1)
        # to range(0, 255)
        r_pred = normalize(r_pred, 0, n_clusters-1, 0, 255)

        plt.imshow(r_pred, cmap='gray')
        plt.show()

    n_clusters += 1






