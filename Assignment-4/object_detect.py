import numpy as np
import cv2 as cv

from hog import extractHogFromImage


def compute_scales(img_shape, window_shape, downscale):
    yield 1
    img_shape = np.array(img_shape)
    window_shape = np.array(window_shape)

    n_scale = 1
    img_shape = img_shape / downscale
    while np.all(window_shape < img_shape):
        yield downscale ** n_scale
        img_shape /= downscale
        n_scale += 1


def sliding_window(image, scale, window_shape, window_stride):
    
    n_rows, n_cols = window_shape[0] * scale, window_shape[1] * scale
    row_step, col_step = window_stride
    
    for row in range(0, image.shape[0] - n_rows, row_step):
        for col in range(0, image.shape[1] - n_cols, col_step):
            patch = image[row: row + n_rows, col: col + n_cols]
            
            if scale != 1:
                patch = cv.resize(patch, window_shape)

            yield row, col, patch

def multi_scale_detector(img, classifier, downscale, window_shape, window_stride, block_shape, block_stride, cell_shape):

    hog = []
    row_col_scale = []

    for scale in compute_scales(img.shape, window_shape, downscale):
        for row, col, window in sliding_window(img, scale, window_shape, window_stride):
            img_hog = extractHogFromImage(window, block_shape, block_stride, cell_shape)
            hog.append(img_hog)
            row_col_scale.append((row, col, scale))
    
    hog = np.array(hog)
    print(hog.shape)
    row_col_scale = np.array(row_col_scale)

    if hog.size > 0:
        labels = classifier.predict(hog)

        bounding_boxes = np.array(
            [
                (col, row, col + window_shape[1] * scale, row + window_shape[0] * scale)
                for row, col, scale in row_col_scale[labels == 1]
            ]
        )
    else:
        return np.array([]), np.array([])

    if hog[labels == 1].size > 0:
        scores = classifier.decision_function(hog[labels == 1])
    else:
        scores = []

    return bounding_boxes, np.array(scores)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, scores, overlapThresh):
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
    idxs = np.argsort(scores)

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


def detect_face(img, classifier, downscale=2, window_shape=(36, 36), window_stride=(6, 6), block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6)):

    # Detect faces
    # Bounding boxes
    bounding_boxes, scores = multi_scale_detector(img, classifier, downscale, window_shape, window_stride, block_shape, block_stride, cell_shape)
    
    # apply non-maximum-suppression
    if bounding_boxes.size > 0 and scores.size > 0:
        detections = non_max_suppression_fast(bounding_boxes, scores, 0)
    else:
        detections = np.array([])
    
    return detections, bounding_boxes

