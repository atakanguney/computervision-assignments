# %%
import os
from importlib import reload

import cv2 as cv

from hog import extractHogFromImage, extractHogFromRandomCrop
from classifier_train import train_classifier, extract_features_negative_images, extract_features_positive_images
from object_detect import detect_face
from evaluation import mean_intersection_over_union, average_precision, create_detections_dict
from loadTestsGT_updated import loadTestsGT
from visualization import visualize_all


def get_img_paths(folder):
    assert os.path.exists(folder), "{} is not found".format(folder)

    img_names = os.listdir(folder)
    return [os.path.join(folder, img_name) for img_name in img_names if img_name.endswith(".jpg")]


if __name__ == "__main__":

    NON_FACE_IMAGES_FOLDER = "data/NonFaceImages/"
    FACE_IMAGES_FOLDER = "data/FaceImages/"
    VALIDATION_IMAGES_FOLDER = "data/ValidationSet/"

    positive_image_paths = get_img_paths(FACE_IMAGES_FOLDER)
    negative_image_paths = get_img_paths(NON_FACE_IMAGES_FOLDER)
    validation_image_paths = get_img_paths(VALIDATION_IMAGES_FOLDER)

    positive_features = extract_features_positive_images(positive_image_paths)
    negative_features = extract_features_negative_images(negative_image_paths, window_shape=(36, 36), window_stride=(48, 48), block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6))
    classifier = train_classifier(positive_features, negative_features)

    window_shape = (72, 72)
    window_stride = (8, 8)
    block_shape = (24, 24)
    block_stride = (12, 12)
    cell_shape = (12, 12)

    detections_dict3 = {}
    for img_path in validation_image_paths:
        print(img_path)
        img_name = os.path.split(img_path)[1]
        detections, bbs = detect_face(cv.imread(img_path, 0), classifier, window_shape=window_shape, window_stride=window_stride, block_shape=block_shape, block_stride=block_stride, cell_shape=cell_shape)
        if detections.size > 0:
            detections_dict3[img_name] = detections

    grand_truths = loadTestsGT()
    grand_truths_dict = {}
    for img in grand_truths:
        grand_truths_dict.setdefault(img[0], []).append(img[1:])

    print("Mean IoU: {}".format(mean_intersection_over_union(detections_dict3, grand_truths_dict)))
    print("Average Precision: {}".format(average_precision(detections_dict3, grand_truths_dict, 0.1)))

    valid_image_paths_dict = {}
    for img_path in validation_image_paths:
        img_name = os.path.split(img_path)[1]
        valid_image_paths_dict[img_name] = img_path

    visualize_all(valid_image_paths_dict, grand_truths_dict, detections_dict3, "results_36_48_12_6_6-2")
