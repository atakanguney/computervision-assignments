# %%
import os
from importlib import reload

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

def initiate_desc(win_size=(36, 36), block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):
    winSize = win_size
    blockSize = block_size
    blockStride = block_stride
    cellSize = cell_size
    nbins = 9

    hog_desc = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    return hog_desc    
# %%
def opencv_classifier(win_size=(36, 36), block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):

    hog_desc = initiate_desc(win_size=win_size,
                             block_size=block_size,
                             block_stride=block_stride,
                             cell_size=cell_size)
    winStride = (48, 48)
    
    def opencv_extract_features(positive_image_paths, negative_features):
        positive_features = [hog_desc.compute(cv.imread(img_path, 0)).reshape(900) for img_path in positive_image_paths]
        negative_features = []
        for img_path in negative_image_paths:
            negative_features.extend(hog_desc.compute(cv.imread(img_path, 0), winStride).reshape(-1, 900))

        return positive_features, negative_features 

    positive_features_cv, negative_features_cv = opencv_extract_features(positive_image_paths, negative_image_paths)
    classifier_cv = train_classifier(positive_features_cv, negative_features_cv)

    return classifier_cv, positive_features_cv, negative_features_cv


# %%


# if __name__ == "__main__":

positive_image_paths = get_img_paths(FACE_IMAGES_FOLDER)
negative_image_paths = get_img_paths(NON_FACE_IMAGES_FOLDER)
validation_image_paths = get_img_paths(VALIDATION_IMAGES_FOLDER)

# %%
import hog
reload(hog)
from hog import extractHogFromImage, extractFromRandomCrop
# %%
# Extract HOG features for positive and negative examples
positive_features, negative_features = extract_features(positive_image_paths, negative_image_paths)

# %%
import classifier_train
reload(classifier_train)
from classifier_train import train_classifier
# %%
# Train classifier with positive and negative samples
classifier = train_classifier(positive_features, negative_features, test=True)

# %%
classifier_cv, positive_features_cv, negative_features_cv = opencv_classifier()

# %%
import object_detect
reload(object_detect)
from object_detect import detect_faces

# %%
# Detect faces in images from validation data set
detections, bounding_boxes = detect_faces(validation_image_paths, classifier, hog.hog_features, opencv=False)

# %%
detections

# %%
detections_dict = create_detections_dict(VALIDATION_IMAGES_FOLDER, detections)

validation_gt = loadTestsGT()
valid_gt_dict = {}
for gt_val in validation_gt:
    valid_gt_dict.setdefault(gt_val[0], []).append(gt_val[1:])

# %%
# valid_gt_dict
detections_dict

# %%
import evaluation
reload(evaluation)
from evaluation import mean_intersection_over_union, average_precision

# %%
# Evaluate detected faces
print("Mean IoU: {}".format(mean_intersection_over_union(detections_dict, valid_gt_dict)))
print("Average Precision: {}".format(average_precision(detections_dict, valid_gt_dict)))

# %%
import visualization
reload(visualization)
from visualization import visualize

# %%
sample_img_name = "addams-family.jpg"
sample_img = cv.imread(VALIDATION_IMAGES_FOLDER + sample_img_name, 0)
# visualize(sample_img, valid_gt_dict[sample_img_name], detections_dict[sample_img_name])
sample_img.shape

# %%
h_ = hog.hog_features(img)
h_.shape
# %%
mag, ang = hog.compute_gradients(sample_img)
mag.shape

# %%
window_mag, window_ang = hog._divide_windows(mag, ang, (36, 36), (6, 6))
window_mag.shape

# %%
np.all(mag[6:42, :36] == window_mag[1, 0])

# %%
block_mag, block_ang = hog._divide_blocks(window_mag, window_ang, (12, 12), (6, 6))
block_mag.shape

# %%
window_mag[0, 0, :12, :12]
# %%
block_mag[0, 0, 1, 0]
# %%
np.all(window_mag[0, 1, 12:24, 6:18] == block_mag[0, 1, 2, 1])
# %%
cell_mag, cell_ang = hog._divide_cells(block_mag, block_ang, (6, 6))
cell_mag.shape

# %%
import numpy as np
# %%
np.all(mag[6:18, 6:18] == block_mag[1][1])

# %%
block_mag.strides
# %%
cell_mag[0, 0, 0, 1]

# %%
block_mag[0][0]
# %%
np.all(block_mag[0, 0, 0, 3, 6:12, -6:] == cell_mag[0, 0, 0, 3, 1, -1])

# %%
img = cv.imread(positive_image_paths[0], 0)
# %%
img = np.float64(img) / 255.0

# %%
import timeit

timeit.timeit("from __main__ import hog, img; hog.hog_features(img)", number=1)
# %%
h_ = hog.hog_features(img)
h_.shape

# %%
hog_desc = initiate_desc()
# %%
h__ = hog_desc.compute(img)

# %%
h_.dtype
