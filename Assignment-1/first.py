import collections
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Path of the image
img_path = "data/Images/Original Images/img_001.jpg"
img = cv.imread(img_path)

# Split image into its channels
blue, green, red = cv.split(img)

# Convert BGR image into RGB to plot images properly
def bgr2rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


plt.subplot("231"), plt.imshow(bgr2rgb(img)), plt.title("ORIGINAL")
plt.subplot("232"), plt.imshow(blue), plt.title("BLUE")
plt.subplot("233"), plt.imshow(green), plt.title("GREEN")
plt.subplot("234"), plt.imshow(red), plt.title("RED")
plt.show()

# Convert BGR image into HSV color space
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# Split HSV color space image into its channels
h, s, v = cv.split(hsv_img)


def normalize(img_, start, end):
    factor = end - start
    img_max = img_.max()
    img_min = img_.min()
    return start + (factor/(img_max - img_min)) * (img_ - img_min)

# Map Hue, Saturation and Value channels to range (0, 255)
h = normalize(h, 0, 255)
s = normalize(s, 0, 255)
v = normalize(v, 0, 255)

plt.subplot("231"), plt.imshow(bgr2rgb(img)), plt.title("ORIGINAL")
plt.subplot("232"), plt.imshow(h), plt.title("HUE")
plt.subplot("233"), plt.imshow(s), plt.title("SATURATION")
plt.subplot("234"), plt.imshow(v), plt.title("VALUE")
plt.show()


# Histogram function
def histogram(img, n_bins=256):
    bins = np.arange(0, n_bins + 1)
    dict_ = collections.Counter(np.digitize(img, bins).flatten())

    # Return x axis values and height values to plot bar
    return list(dict_.keys()), list(dict_.values())


images = {
    'RED': red,
    'GREEN': green,
    'BLUE': blue,
    'HUE': h,
    'SATURATION': s,
    'Value': v
}

for title in images:
    x, height = histogram(images[title])
    plt.title(title)
    plt.bar(x, height=height)
    plt.show()
