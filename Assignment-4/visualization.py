import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def visualize(img, true_bb, pred_bb):
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img)

    for bb in true_bb:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor="r",
                                 facecolor="none")
        ax.add_patch(rect)

    for bb in pred_bb:
        if not bb:
            continue
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor="lime",
                                 facecolor="none")
        ax.add_patch(rect)

    plt.show()


def visualize_all(image_paths, true_bbs, pred_bbs):
    for i, img_name in enumerate(true_bbs):
        img = cv.imread(image_paths[i])
        true_bb = true_bbs[img_name]
        pred_bb = pred_bbs[img_name]

        visualize(img, true_bb, pred_bb)
