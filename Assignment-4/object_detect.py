import numpy as np
import cv2 as cv
from sklearn.feature_extraction.image import extract_patches


def pyramid(image, scale=2, minSize=(36, 36)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = cv.resize(image, (0, 0), fx=1.0 / scale, fy=1.0 / scale)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(img, window_size, window_stride):
    n_rows, n_cols = img.shape
    s_row, s_col = window_stride
    w_row, w_col = window_size

    def _check_window_shape(shape1, shape2):
        return shape1[0] == shape2[0] and shape1[1] == shape2[1]

    for row in range(0, n_rows, s_row):
        for col in range(0, n_cols, s_col):
            window = img[row: row + w_row, col: col + w_col]

            if _check_window_shape(window.shape, window_size):
                yield (row, col, window)


def multi_scale_detector(img, classifier, downscale, win_size, win_stride, hog_features, opencv=False):

    bounding_boxes = []

    scale = 1

    for resized in pyramid(img, scale=2, minSize=win_size):

        for row, col, window_img in sliding_window(resized, win_size, win_stride):
            if opencv:
                hog_ = hog_features(window_img.astype(np.uint8)).reshape(900)
            else:
                hog_ = hog_features(window_img)

            if classifier.predict([hog_]):
                rect = np.array(
                    [row, col, row + win_size[0], col + win_size[1]])
                rect *= scale
                bounding_boxes.append(rect)

        scale *= 2

    return np.array(bounding_boxes)


def get_images(image_paths, opencv=False):
    images = [cv.imread(img_path, 0) for img_path in image_paths]
    if not opencv:
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


def detect_faces(image_paths, classifier, hog_features, downscale=2, win_size=(36, 36), win_stride=(6, 6), opencv=False):

    # Get images
    images = get_images(image_paths, opencv=opencv)

    # Detect faces
    # Bounding boxes
    bounding_boxes_all = [multi_scale_detector(
        img, classifier, downscale, win_size, win_stride, hog_features, opencv=opencv) for img in images]

    # apply non-maximum-suppression
    detections_all = [non_max_suppression_fast(
        bounding_boxes, 0) for bounding_boxes in bounding_boxes_all]

    return detections_all, bounding_boxes_all
