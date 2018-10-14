from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def obtain_skin_color_mask(img, bin_mask):
    skin_colors = img[bin_mask == 255, :]
    non_skin_colors = img[bin_mask == 0, :]

    color = np.vstack((skin_colors, non_skin_colors))
    target = np.concatenate((np.ones(len(skin_colors)), np.zeros(len(non_skin_colors))))

    train , test, train_target, test_target = train_test_split(color, target, test_size=0.33)

    logistic = LogisticRegressionCV()
    logistic.fit(train, train_target)

    score = logistic.score(test, test_target)
    print("Test Results {}".format(score))

    color_vectors = img.reshape(-1, 3)
    predict_skin = logistic.predict(color_vectors).reshape(img.shape[:-1])

    return predict_skin

if __name__ == "__main__":
    bin_masks = np.load("masks.npy")
    skin_color_masks = []

    # Image path
    imgs_base = "data/Images/Original Images/"
    n_imgs = 20
    imgs = [cv.imread(imgs_base + "img_{0:03d}.jpg".format(i)) for i in range(1, n_imgs + 1)]

    for i, bin_mask in enumerate(bin_masks):
        skin_color_mask = obtain_skin_color_mask(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB), bin_mask)
        skin_color_masks.append(skin_color_mask)

    for i, color_mask in enumerate(skin_color_masks):
        plt.imshow(color_mask)
        #plt.title("Image {0:03d}".format(i+1))
        plt.savefig("Logistic Regression Results/skin_color_mask_{0:03d}".format(i+1))
        plt.show()