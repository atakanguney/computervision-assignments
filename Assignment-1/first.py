from collections import Counter
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
plt.colorbar()
plt.subplot2grid(layout, (1, 1))
plt.imshow(green), plt.title("GREEN")
plt.colorbar()
plt.subplot2grid(layout, (1, 2))
plt.imshow(red), plt.title("RED")
plt.colorbar()
plt.savefig("Task 1 Plots/RGB_plots.png")
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
#plt.imshow(bgr2rgb(img)), plt.title("ORIGINAL")
plt.imshow(hsv_img), plt.title("HSV")
plt.subplot2grid(layout, (1, 0))
plt.imshow(h), plt.title("HUE")
plt.colorbar()
plt.subplot2grid(layout, (1, 1))
plt.imshow(s), plt.title("SATURATION")
plt.colorbar()
plt.subplot2grid(layout, (1, 2))
plt.imshow(v), plt.title("VALUE")
plt.colorbar()
plt.savefig("Task 1 Plots/HSV_plots.png")
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
    bins = np.linspace(range[0], range[1], n_bins + 1)

    bin_indices = np.searchsorted(bins, seq, side='left')
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
    # plt.subplot("221")
    plt.title(title) # + " mine")
    plt.bar(bins_m[:-1], height=hist_m)
    plt.savefig("Task 1 Plots/histogram_{}".format(title))
    plt.show()
    # plt.subplot("222")
    hist, bins = np.histogram(images[title].ravel(), 256, [0, 256])
    assert np.equal(hist, hist_m).all(), "My histogram function is not equal to numpy histogram"
    # plt.bar(bins[:-1], height=hist)
    # plt.title(title)

    # plt.subplot("223")
    # plt.title("BINCOUNT")
    hist_b = np.bincount(images[title].flatten(), minlength=256)
    assert np.equal(hist, hist_b).all(), "My histogram function is not equal to bincount"
    # plt.bar(range(0, 256), height=hist_b)
    # plt.subplot("224")
    # plt.title("COUNTER")

    def counter(seq):
        z = Counter(seq.tolist())
        hist = np.zeros(256)
        for el in z:
            hist[el] = z[el]

        bins = np.arange(0, 256)

        return hist, bins

    hist_c, bins_c = counter(images[title].ravel())
    assert np.equal(hist, hist_c).all(), "My histogram function is not equal to counter"
    # plt.bar(bins_c, height=hist_c)
    # plt.show()

