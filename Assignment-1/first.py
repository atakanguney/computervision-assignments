import collections
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Path of the image
img_path = "data/Images/Original Images/img_001.jpg"
img = cv.imread(img_path, 1)

# Split image into its channels
blue, green, red = cv.split(img)


# Convert BGR image into RGB to plot images properly
def bgr2rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


layout = (2, 3)
plt.subplot2grid(layout, (0, 1), colspan=1)
plt.imshow(bgr2rgb(img)), plt.title("ORIGINAL")
plt.subplot2grid(layout, (1, 0))
plt.imshow(blue), plt.title("BLUE")
plt.subplot2grid(layout, (1, 1))
plt.imshow(green), plt.title("GREEN")
plt.subplot2grid(layout, (1, 2))
plt.imshow(red), plt.title("RED")
plt.show()

# Convert BGR image into HSV color space
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# Split HSV color space image into its channels
h, s, v = cv.split(hsv_img)


def normalize(seq, input_start, input_end, output_start, output_end):
    factor = (output_end - output_start) / (input_end - input_start)
    return (output_start + factor * (seq - input_start)).astype(np.uint8)


# Map Hue channel to range (0, 255)
h = normalize(h, 0, 179, 0, 255)

layout = (2, 3)
plt.subplot2grid(layout, (0, 1), colspan=1)
plt.imshow(bgr2rgb(img)), plt.title("ORIGINAL")
plt.subplot2grid(layout, (1, 0))
plt.imshow(h), plt.title("HUE")
plt.subplot2grid(layout, (1, 1))
plt.imshow(s), plt.title("SATURATION")
plt.subplot2grid(layout, (1, 2))
plt.imshow(v), plt.title("VALUE")
plt.show()


def find_bin(num, bins):
    n_bins = len(bins) - 1

    for i in range(n_bins):
        if (num >= bins[i]) and (num < bins[i + 1]):
            return i

    return None


def bin_numbers(seq, bins):
    binnumbers = []
    for num in seq:
        bin_number = find_bin(num, bins)

        if not bin_number:
            raise ValueError("Couldn't find bin number or element {}".format(num))

        binnumbers.append(bin_number)

    return np.array(binnumbers)


# Histogram function
def histogram(seq, n_bins=256, range=(0, 256)):
    temp_seq = seq.copy()
    temp_seq = temp_seq[(temp_seq < range[1]) & (temp_seq >= range[0])]
    bins = np.linspace(range[0], range[1], n_bins + 1)
    bin_indices = np.concatenate((
        bins.searchsorted(temp_seq[:-1], side='left'),
        bins.searchsorted(temp_seq[-1:], side='right')
    ))
    hist = np.bincount(bin_indices, minlength=n_bins)

    # Return x axis values and height values to plot bar
    return hist, bins


images = {
    'RED': red,
    'GREEN': green,
    'BLUE': blue,
    'HUE': h,
    'SATURATION': s,
    'Value': v
}

for title in images:
    hist_m, bins_m = histogram(images[title].ravel(), 256, [0, 256])
    plt.subplot("121")
    plt.title(title + ' mine')
    plt.bar(bins_m[:-1], height=hist_m)
    plt.subplot("122")
    hist, bins = np.histogram(images[title].ravel(), 256, [0, 256])
    plt.bar(bins[:-1], height=hist)
    plt.title(title)
    plt.show()
