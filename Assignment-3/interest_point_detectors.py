#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:42:22 2018

@author: Atakan Guney
"""
# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# %%
# Harris Corner Detector
def myHarrisCornerDetector(image, ksize=5, k=0.04, threshold=500):
    # Compute x and y derivatives
    derivatives_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=ksize)
    derivatives_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=ksize)

    # Compute products of derivatives at every pixel
    derivatives_xx = derivatives_x * derivatives_x
    derivatives_yy = derivatives_y * derivatives_y
    derivatives_xy = derivatives_x * derivatives_y
    # Compute the sums of products of derivatives at each pixel
    s_xx = cv.GaussianBlur(derivatives_xx, (ksize, ksize), 1)
    s_yy = cv.GaussianBlur(derivatives_yy, (ksize, ksize), 1)
    s_xy = cv.GaussianBlur(derivatives_xy, (ksize, ksize), 1)
    # Define at each pixel
        # H(x, y)
    # Compute the response of the detector at each pixel
    det = s_xx * s_yy - s_xy * s_xy
    trace = s_xx + s_yy
    r = det - k * (trace * trace)
    print("MAX R: {}".format(r.max()))
    print("MIN R: {}".format(r.min()))
    # Threshold on value of R. Compute nonmax supression
    return np.where(r > threshold)

# %%
kuzey_path = "Images/kuzey.jpg"

kuzey = cv.imread(kuzey_path, 0) * (1 / 255.0)
#plt.imshow(kuzey, cmap="gray")

# for threshold in range(100, 300, 25):
#     x, y = myHarrisCornerDetector(kuzey, threshold=threshold)
#     print("Detected Corners: {}".format(len(x)))
x, y = myHarrisCornerDetector(kuzey)

kuzey_bgr = cv.imread(kuzey_path)
kuzey_rgb = cv.cvtColor(kuzey_bgr, cv.COLOR_BGR2RGB)

plt.imshow(kuzey_rgb)
plt.plot(y, x, "xy", markersize=1)
plt.show()

# %%
# Preprocessing
qualities = [
    [int(cv.IMWRITE_JPEG_QUALITY), 5],
    [int(cv.IMWRITE_JPEG_QUALITY), 25],
    [int(cv.IMWRITE_JPEG_QUALITY), 45],
    [int(cv.IMWRITE_JPEG_QUALITY), 65],
    [int(cv.IMWRITE_JPEG_QUALITY), 85],
    [int(cv.IMWRITE_JPEG_QUALITY), 105],     
]

# %%
for quality in qualities:
    cv.imwrite("kuzey-{}.jpg".format(quality[1]), kuzey_bgr, quality)

# %%
variances = [4 ** i for i in range(1, 7)]

# %%
img_shape = kuzey_rgb.shape
for variance in variances:
    noise = np.random.rand(*img_shape) * variance
    cv.imwrite("kuzey-var-{}.jpg".format(variance), kuzey_bgr + noise)

# %%
# Measuring repeatibility

def measureRepeatibility(keyPoints1, keyPoints2, homogrophy1to2, image2size):
    pass