import numpy as np
import cv2 as cv

from skimage.transform import pyramid_gaussian
from sklearn.feature_extraction.image import extract_patches

from hog import hog_features


def multi_scale_detector(img, classifier, downscale, win_size, win_stride):

    bounding_boxes = []
    scale = 1

    for resized in pyramid_gaussian(img, downscale=downscale, multichannel=False):

        print(resized.shape)
        if resized.shape[0] < win_size[0] or resized.shape[1] < win_size[1]:
            print("Break occured")
            break

        windows = extract_patches(resized, patch_shape=win_size, extraction_step=win_stride)

        for row in range(windows.shape[0]):
            for col in range(windows.shape[1]):
                hog_ = hog_features(windows[row][col])
                print(hog_.shape)
                print(classifier.predict(hog_.reshape(1, -1)))
                if classifier.predict(hog_.reshape(1, -1)):
                    print("Face detected")
                    x1 = row * win_stride[0] * scale
                    y1 = col * win_stride[1] * scale
                    x2 = x1 + win_size[0] * scale
                    y2 = y1 + win_size[1] * scale
                    bounding_boxes.append([x1, y1, x2, y2])

        scale *= 2

    return np.array(bounding_boxes)


def get_images(image_paths):
    images = [cv.imread(img_path, 0) for img_path in image_paths]
    images = [np.float64(img) / 255.0 for img in images]

    return images


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def detect_faces(image_paths, classifier, downscale=2, win_size=(36, 36), win_stride=(6, 6)):

    # Get images
    images = get_images(image_paths)

    # Detect faces
    # Bounding boxes
    bounding_boxes_all = [multi_scale_detector(img, classifier, downscale, win_size, win_stride) for img in images]

    # apply non-maximum-suppression
    detections_all = [non_max_suppression_fast(bounding_boxes, 0) for bounding_boxes in bounding_boxes_all]

    return detections_all, bounding_boxes_all
