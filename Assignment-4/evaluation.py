import itertools as it

import numpy as np


def create_detections_dict(folder, detections):
    assert os.path.exists(folder), "{} is not found".format(folder)

    img_names = os.listdir(folder)
    res_dict = {}
    for i, name in enumerate(img_names):
        res_dict[name] = detections[i]

    return res_dict


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def mean_intersection_over_union(predicted_bbs, true_bbs):
    iou = []
    for img in predicted_bbs:
        if img not in true_bbs:
            continue
        for true, pred in it.product(true_bbs[img], predicted_bbs[img]):
            if pred.size != 0:
                iou.append(bb_intersection_over_union(true, pred))

    iou = np.array(iou)
    return iou[iou > 0].mean()


def average_precision(predicted_bbs, true_bbs, threshold=0.5):
    tp = 10**-8
    fp = 10**-8
    fn = 10**-8

    precision_sum = 0
    for img in predicted_bbs:
        if img not in true_bbs:
            continue
        for true, pred in it.product(true_bbs[img], predicted_bbs[img]):
            if pred.size == 0:
                fn += 1
                continue
            iou = bb_intersection_over_union(true, pred)
            if iou > threshold:
                tp += 1
            elif iou > 0:
                fp += 1

        precision = tp / (tp + fp)
        precision_sum += precision

    return precision_sum / len(predicted_bbs)
