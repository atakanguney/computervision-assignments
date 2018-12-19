# %%
import os

import cv2 as cv

from hog import extractHogFromImage, extractFromRandomCrop
from classifier_train import train_classifier
from object_detect import detect_faces
from evaluation import mean_intersection_over_union, average_precision
from loadTestsGT_updated import loadTestsGT
from visualization import visualize
# %%
NON_FACE_IMAGES_FOLDER = "data/NonFaceImages/"
FACE_IMAGES_FOLDER = "data/FaceImages/"
VALIDATION_IMAGES_FOLDER = "data/SampleValidationSet/"
# %%


def get_img_paths(folder):
    assert os.path.exists(folder), "{} is not found".format(folder)

    img_names = os.listdir(folder)
    if folder.endswith("/"):
        return [folder + img_name for img_name in img_names if img_name.endswith(".jpg")]
    else:
        return [folder + "/" + img_name for img_name in img_names if img_name.endswith(".jpg")]


def extract_features(positive_image_paths, negative_image_paths):
    positive_features = [extractHogFromImage(img_path) for img_path in positive_image_paths]
    negative_features = []
    for img_path in negative_image_paths:
        negative_features.extend(extractFromRandomCrop(img_path))

    return positive_features, negative_features
# %%


def create_detections_dict(folder, detections):
    assert os.path.exists(folder), "{} is not found".format(folder)

    img_names = os.listdir(folder)
    res_dict = {}
    for i, name in enumerate(img_names):
        res_dict[name] = detections[i]

    return res_dict

# %%


# if __name__ == "__main__":

positive_image_paths = get_img_paths(FACE_IMAGES_FOLDER)
negative_image_paths = get_img_paths(NON_FACE_IMAGES_FOLDER)
validation_image_paths = get_img_paths(VALIDATION_IMAGES_FOLDER)

# %%
# Extract HOG features for positive and negative examples
positive_features, negative_features = extract_features(positive_image_paths, negative_image_paths)

# %%
# Train classifier with positive and negative samples
classifier = train_classifier(positive_features, negative_features)

# # %%
# # Detect faces in images from validation data set
# detections = detect_faces(validation_image_paths, classifier)
# # %%
# detections_dict = create_detections_dict(VALIDATION_IMAGES_FOLDER, detections)
#
# validation_gt = loadTestsGT()
# valid_gt_dict = {}
# for gt_val in validation_gt:
#     valid_gt_dict.setdefault(gt_val[0], []).append(gt_val[1:])
#
# # %%
# # Evaluate detected faces
# print("Mean IoU: {}".format(mean_intersection_over_union(detections_dict, valid_gt_dict)))
# print("Average Precision: {}".format(average_precision(detections_dict, valid_gt_dict)))
#
# # %%
# sample_img_name = "baseball.jpg"
# sample_img = cv.imread(VALIDATION_IMAGES_FOLDER + sample_img_name)
# visualize(sample_img, valid_gt_dict[sample_img_name], detections_dict[sample_img_name])
