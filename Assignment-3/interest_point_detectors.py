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
from scipy.spatial.distance import cdist
# %%
def non_max_suppression(matrix, rows, cols, nms_size):
    corners = []
    for row, col in zip(rows, cols):
        # Find lower and upper bounds for row
        row_low = max(0, row - nms_size)
        row_up = min(matrix.shape[0], row + nms_size)
        
        # Find lower and upper bounds for column
        col_low = max(0, col - nms_size)
        col_up = min(matrix.shape[1], col + nms_size)
        
        # Find local max
        local_matrix = matrix[row_low:row_up, col_low:col_up]
        #local_max = local_matrix.max()
        local_max_arg = np.unravel_index(local_matrix.argmax(), local_matrix.shape)
        
        row_max = row_low + local_max_arg[0]
        col_max = col_low + local_max_arg[1]
        
        if not (row_max, col_max) in corners:
            corners.append((row_max, col_max))
     
    corners = np.array(corners)
    
    return corners[:, 0], corners[:, 1]


def normalize(matrix, new_min, new_max):
    min_ = np.min(matrix)
    max_ = np.max(matrix)
    
    slope = (new_max - new_min) / (max_ - min_)
    return (matrix - min_) * slope + new_min


# Harris Corner Detector
def myHarrisCornerDetector(image, ksize=5, k=0.04, threshold=None, nms_size=5):
    # Compute x and y derivatives
    derivatives_x = cv.Sobel(image, cv.CV_64FC1, 1, 0, ksize=ksize)
    derivatives_y = cv.Sobel(image, cv.CV_64FC1, 0, 1, ksize=ksize)
    # Compute products of derivatives at every pixel
    derivatives_xx = derivatives_x ** 2
    derivatives_yy = derivatives_y ** 2
    derivatives_xy = derivatives_x * derivatives_y
    # Compute the sums of products of derivatives at each pixel
    s_xx = cv.GaussianBlur(derivatives_xx, (ksize, ksize), 1, 0)
    s_yy = cv.GaussianBlur(derivatives_yy, (ksize, ksize), 0, 1)
    s_xy = cv.GaussianBlur(derivatives_xy, (ksize, ksize), 1, 1)
    # Define at each pixel
        # H(x, y)
    # Compute the response of the detector at each pixel
    det = s_xx * s_yy - s_xy * s_xy
    trace = s_xx + s_yy
    r = det - k * (trace ** 2)
    # print("MAX R: {}".format(r.max()))
    # print("MIN R: {}".format(r.min()))
    
    # TODO: inspect normalizing r values
    # r = normalize(r, 0, 255)
    
    # print("MAX R: {}".format(r.max()))
    # print("MIN R: {}".format(r.min()))

    # Threshold on value of R
    if not threshold:
        #threshold = r.max() * 0.01
        threshold = r.max() * 0.01
        
    y, x = np.where(r > threshold)

    # Compute nonmax supression
    y, x = non_max_suppression(r, y, x, nms_size)
    
    return x, y

# %%
kuzey_path = "Images/kuzey.jpg"

kuzey = cv.imread(kuzey_path, 0)
#plt.imshow(kuzey, cmap="gray")

# for threshold in range(100, 300, 25):
#     x, y = myHarrisCornerDetector(kuzey, threshold=threshold)
#     print("Detected Corners: {}".format(len(x)))
x, y = myHarrisCornerDetector(kuzey)
kuzey_bgr = cv.imread(kuzey_path)
kuzey_rgb = cv.cvtColor(kuzey_bgr, cv.COLOR_BGR2RGB)

plt.imshow(kuzey_rgb)
plt.plot(x, y, "xy", markersize=2)
plt.show()

# %%
# Preprocessing - 1
def preprocessin_part1(qualities=None):
    kuzey_path = "Images/kuzey.jpg"
    kuzey_bgr = cv.imread(kuzey_path)

    if not qualities:
        qualities = [
            [int(cv.IMWRITE_JPEG_QUALITY), 5],
            [int(cv.IMWRITE_JPEG_QUALITY), 25],
            [int(cv.IMWRITE_JPEG_QUALITY), 45],
            [int(cv.IMWRITE_JPEG_QUALITY), 65],
            [int(cv.IMWRITE_JPEG_QUALITY), 85],
            [int(cv.IMWRITE_JPEG_QUALITY), 105],     
        ]

    for quality in qualities:
        cv.imwrite("kuzey-{}.jpg".format(quality[1]), kuzey_bgr, quality)

# %%
# Preprocessing - 2
def preprocessing_part2(variances=None):
    kuzey_path = "Images/kuzey.jpg"
    kuzey_bgr = cv.imread(kuzey_path)
    
    if not variances:
        variances = [4 ** i for i in range(1, 7)] 
    
    img_shape = kuzey_bgr.shape
    for variance in variances:
        noise = np.random.rand(*img_shape) * variance
        cv.imwrite("kuzey-var-{}.jpg".format(variance), kuzey_bgr + noise)

# %%
# Measuring repeatability
def in_image(points, imgsize):
    return (points[0] >= 0) \
        & (points[1] >= 0) \
        & (points[0] < imgsize[1]) \
        & (points[1] < imgsize[0])


def convert_homogenous(points):
    return np.vstack((points, np.ones(points.shape[1])))


def homogenous_matmul(matrix1, matrix2):
    mul = np.matmul(matrix1, matrix2)
    return mul / mul[-1][np.newaxis, :]


def size_epsilon_neighborhood(homography1to2, key_points1, key_points2, epsilon):
    key_points1 = homogenous_matmul(homography1to2, key_points1)
    dists = cdist(key_points1[:-1].T, key_points2[:-1].T)
    return (dists < epsilon).sum()


def measureRepeatability(keyPoints1, keyPoints2, homography1to2, image2size, epsilon=1.5):
    # Convert key points into compatible format
    print("Key points image 1: {}".format(keyPoints1.shape[1]))
    print("Key points image 2: {}".format(keyPoints2.shape[1]))
    keyPoints1 = convert_homogenous(keyPoints1)
    keyPoints2 = convert_homogenous(keyPoints2)
    
    keyPoints1to2 = homogenous_matmul(homography1to2, keyPoints1)
    keyPoints2to1 = homogenous_matmul(np.linalg.pinv(homography1to2), keyPoints2)
    
    common_part_image_1 = keyPoints1to2[:, in_image(keyPoints1to2, image2size)]
    common_part_image_2 = keyPoints2to1[:, in_image(keyPoints2to1, image2size)]
    
    print("Common part image1: {}".format(common_part_image_1.shape[1]))
    print("Common part image2: {}".format(common_part_image_2.shape[1]))
    r = size_epsilon_neighborhood(homography1to2,
                                  common_part_image_1,
                                  common_part_image_2,
                                  epsilon)
    print("R with epsilon: {} = {}".format(epsilon, r))
    size_common1 = common_part_image_1.shape[1]
    size_common2 = common_part_image_2.shape[1]

    return r / min(size_common1, size_common2)

# %%
def read_homography(filename):
    with open(filename, "r") as file:
        homo = []
        for line in file:
            numbers = line.strip().split()
            numbers = [float(number) for number in numbers]
            homo.append(numbers)
            
        return np.array(homo)
        
# %%
# Import images
n_images = 6

images = []
for i in range(1, n_images + 1):
    images.append(cv.cvtColor(cv.imread("Images/img{}.png".format(i)), cv.COLOR_BGR2RGB))

# Find key points with myHarrisCornerDetector
gray_images = [cv.cvtColor(img, cv.COLOR_RGB2GRAY) for img in images]
key_points_harris = [np.array(myHarrisCornerDetector(gray)).squeeze() for gray in gray_images]
# Find key points with SIFT detector
gray_images = [cv.cvtColor(img, cv.COLOR_RGB2GRAY) for img in images]
sift = cv.xfeatures2d.SIFT_create()
key_points_sift = [np.array(list(map(lambda x: x.pt, sift.detect(gray, None)))).squeeze().T for gray in gray_images]
# Find key points with SURF detector
surf = cv.xfeatures2d.SURF_create()
key_points_surf = [np.array(list(map(lambda x: x.pt, surf.detect(gray, None)))).squeeze().T for gray in gray_images]

# %%
# Read homography matrices
homographies = [
    "Images/H1to2p",
    "Images/H1to3p",
    "Images/H1to4p",
    "Images/H1to5p",
    "Images/H1to6p",
]
homos = []
for homo in homographies:
    homos.append(read_homography(homo))
# %%
def plot_harris(img, kp, plt):
    plt.imshow(img)
    plt.plot(kp[0], kp[1], "xy", markersize=2)

# %%
# Measure repeatabilty for each 3 detector
repeatabilities_harris = []
repeatabilities_sift = []
repeatabilities_surf = []

for img in range(1, n_images):
    harris = measureRepeatability(key_points_harris[0], key_points_harris[img], homos[img - 1], gray_images[img].shape, epsilon=1.5)
    sift = measureRepeatability(key_points_sift[0], key_points_sift[img], homos[img - 1], gray_images[img].shape)
    surf = measureRepeatability(key_points_surf[0], key_points_surf[img], homos[img - 1], gray_images[img].shape)
    repeatabilities_harris.append(harris)
    repeatabilities_sift.append(sift)
    repeatabilities_surf.append(surf)
# %%
# Plot the results
images = [i for i in range(1, n_images)]
plt.plot(images, repeatabilities_harris, "-xy", markersize=10)
plt.plot(images, repeatabilities_sift, "-og", markersize=10)
plt.plot(images, repeatabilities_surf, "->k", markersize=10)
plt.legend(["myHarris", "Sift", "Surf"])
plt.xticks(images, ["1-2", "1-3", "1-4", "1-5", "1-6"])
plt.ylabel("Repeatability Rate")
