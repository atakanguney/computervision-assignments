import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def obtain_range(img, binary_mask):
    # Split image into B, G, R
    blue, green, red = cv.split(img)
    # Convert mask image into HSV space
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Get values of each channel
    hue, saturation, value = cv.split(hsv_img)
    # Apply Mask
    red &= binary_mask
    green &= binary_mask
    blue &= binary_mask

    hue &= binary_mask
    saturation &= binary_mask
    value &= binary_mask

    # Obtain range for each channel
    def reduce2range(channel):
        min_ = np.min(channel)
        max_ = np.max(channel)
        return min_, max_

    r_range = reduce2range(red[red > 0].flatten())
    g_range = reduce2range(green[green > 0].flatten())
    b_range = reduce2range(blue[blue > 0].flatten())

    h_range = reduce2range(hue[hue > 0].flatten())
    s_range = reduce2range(saturation[saturation > 0].flatten())
    v_range = reduce2range(value[value > 0].flatten())

    bgr_lower = np.array([b_range[0], g_range[0], r_range[0]])
    bgr_upper = np.array([b_range[1], g_range[1], r_range[1]])

    hsv_lower = np.array([h_range[0], s_range[0], v_range[0]])
    hsv_upper = np.array([h_range[1], s_range[1], v_range[1]])

    return bgr_lower, bgr_upper, hsv_lower, hsv_upper


def predict_skin_color_mask(img, binary_mask):
    bgr_lower, bgr_upper, hsv_lower, hsv_upper = obtain_range(img, binary_mask)
    predicted_mask_BGR = cv.inRange(img, bgr_lower, bgr_upper)
    predicted_mask_HSV = cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, :-1], hsv_lower[:-1], hsv_upper[:-1])

    predicted_mask_combined = cv.bitwise_and(predicted_mask_BGR, predicted_mask_HSV)

    return predicted_mask_combined


def predict_skin_color_masks(imgs, binary_masks):
    predicted_masks_combined = []
    for i, img in enumerate(imgs):
        predicted_mask_combined = predict_skin_color_mask(img, binary_masks[i])
        predicted_masks_combined.append(predicted_mask_combined)

    return predicted_masks_combined


if __name__ == "__main__":
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
    for i, mask in enumerate(masks):
        _, temp_ = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        plt.imshow(temp_, cmap='gray')
        plt.savefig("Task 2 Plots/bin_mask_{0:03d}".format(i + 1))
        plt.show()
        binary_masks.append(temp_)

    predictions = predict_skin_color_masks(imgs, binary_masks)
    for i, pred in enumerate(predictions):
        plt.imshow(pred)
        plt.savefig("Task 2 Plots/skin_color_mask_{0:03d}".format(i + 1))
        plt.show()

    # Erosion
    masks_erosion = []
    kernel = np.ones((5, 5), np.uint8)
    for i, mask in enumerate(binary_masks):
        new_mask = cv.erode(mask, kernel, iterations=1)
        masks_erosion.append(new_mask)
        plt.subplot("121")
        plt.imshow(new_mask, cmap='gray')
        plt.title("with Erosion")
        plt.subplot("122")
        plt.imshow(mask, cmap='gray')
        plt.title("Original")
        plt.savefig("Task 2 Plots/bin_mask_erode_{0:03d}".format(i + 1))
        plt.show()

    predictions_erosion = predict_skin_color_masks(imgs, masks_erosion)
    for i, pred in enumerate(predictions_erosion):
        plt.subplot("121")
        plt.imshow(pred)
        plt.title("with Erosion")
        plt.subplot("122")
        plt.imshow(predictions[i])
        plt.title("w/o Erosion")
        plt.savefig("Task 2 Plots/skin_color_mask_erode_{0:03d}".format(i + 1))
        plt.show()