from functools import reduce

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

n_imgs = 10
imgs = []
masks = []

# Masks and Images base paths
masks_base = "data/Images/Ground Truths/"
imgs_base = "data/Images/Original Images/"

# Read and store images and corresponding masks
for i in range(1, n_imgs + 1):
    img = cv.imread(imgs_base + "img_" + "{0:03d}".format(i) + ".jpg", 1)
    mask = cv.imread(masks_base + "mask_" + "{0:03d}".format(i) + ".jpg", 0)

    imgs.append(img)
    masks.append(mask)

binary_masks = []
for mask in masks:
    _, temp_ = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    binary_masks.append(temp_)

# for mask in masks:
#    plt.imshow(mask, cmap='gray')
#    plt.show()

# rows = []
# cols = []

# for bin_mask in binary_masks:
#    row, col = np.where(bin_mask > 0)
#    rows.append(row)
#    cols.append(col)


def obtain_ranges(imgs, binary_masks):
    # Create arrays for each channel
    reds = []
    greens = []
    blues = []

    hues = []
    saturations = []
    values = []

    for i, img in enumerate(imgs):
        # Split image into B, G, R
        blue, green, red = cv.split(img)
        # Convert mask image into HSV space
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Get values of each channel
        hue, saturation, value = cv.split(hsv_img)
        # Apply Masks
        red &= binary_masks[i]
        green &= binary_masks[i]
        blue &= binary_masks[i]

        hue &= binary_masks[i]
        saturation &= binary_masks[i]
        value &= binary_masks[i]

        # Store values for each channel to corresponding arrays
        reds.append(red[red > 0].flatten())
        blues.append(blue[blue > 0].flatten())
        greens.append(green[green > 0].flatten())

        hues.append(hue[hue > 0].flatten())
        saturations.append(saturation[saturation > 0].flatten())
        values.append(value[value > 0 ].flatten())


    # Obtain ranges for each channel
    def reduce2range(seq):
        min_ = reduce(lambda x, y: min(np.min(x), np.min(y)), seq)
        max_ = reduce(lambda x, y: max(np.max(x), np.max(y)), seq)

        return min_, max_


    r_range = reduce2range(reds)
    g_range = reduce2range(greens)
    b_range = reduce2range(blues)

    h_range = reduce2range(hues)
    s_range = reduce2range(saturations)
    v_range = reduce2range(values)

    bgr_lower = np.array([b_range[0], g_range[0], r_range[0]])
    bgr_upper = np.array([b_range[1], g_range[1], r_range[1]])

    hsv_lower = np.array([h_range[0], s_range[0], v_range[0]])
    hsv_upper = np.array([h_range[1], s_range[1], v_range[1]])

    return bgr_lower, bgr_upper, hsv_lower, hsv_upper


def predict_skin_color_masks(imgs, binary_masks):
    bgr_lower , bgr_upper, hsv_lower, hsv_upper = obtain_ranges(imgs, binary_masks)
    predicted_masks_BGR = [cv.inRange(img, bgr_lower, bgr_upper) for img in imgs]
    predicted_masks_HSV = [cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), hsv_lower, bgr_upper) for img in imgs]

    predicted_masks_combined = [cv.bitwise_and(predicted_masks_BGR[i], predicted_masks_HSV[i]) for i in
                                range(len(predicted_masks_BGR))]

    return predicted_masks_combined

predictions = predict_skin_color_masks(imgs, binary_masks)
for i, pred in enumerate(predictions):
    plt.imshow(pred, cmap='gray')
    plt.title("Image {}".format(i + 1))
    plt.show()

# Erosion

masks_erosion = []
kernel = np.ones((5, 5), np.uint8)
for mask in binary_masks:
    new_mask = cv.erode(mask, kernel, iterations=1)
    masks_erosion.append(new_mask)
    plt.subplot("121")
    plt.imshow(new_mask, cmap='gray')
    plt.title("Erosion")
    plt.subplot("122")
    plt.imshow(mask, cmap='gray')
    plt.title("Original")
    plt.show()

predictions_erosion = predict_skin_color_masks(imgs, masks_erosion)
for i, pred in enumerate(predictions_erosion):
    plt.subplot("121")
    plt.imshow(pred, cmap='gray')
    plt.title("with Erosion")
    plt.subplot("122")
    plt.imshow(predictions[i], cmap='gray')
    plt.title("w/o Erosion")
    plt.show()