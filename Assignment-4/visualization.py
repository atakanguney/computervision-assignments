import os
import cv2 as cv
import matplotlib.pyplot as plt

def plot_bounding_boxes(img, boundind_boxes, color, ax):
    ax.imshow(img)
    for x, y, xx, yy in boundind_boxes:
        ax.add_patch(plt.Rectangle((x, y), xx - x, yy - y, linewidth=1, edgecolor=color,
                                   facecolor="none"))


def visualize_and_save(img, true_bb, pred_bb, img_name, save_file):
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img)

    plot_bounding_boxes(img, true_bb, "r", ax)

    plot_bounding_boxes(img, pred_bb, "lime", ax)

    plt.savefig(os.path.join(save_file, img_name))
    plt.show()


def visualize_all(image_paths_dict, true_bbs, pred_bbs, save_file):
    if not os.path.exists(save_file):
        os.mkdir(save_file)

    for i, img_name in enumerate(pred_bbs):
        if img_name not in true_bbs:
            continue

        img = cv.imread(image_paths_dict[img_name])
        true_bb = true_bbs[img_name]
        pred_bb = pred_bbs[img_name]

        visualize_and_save(img, true_bb, pred_bb, img_name, save_file)
