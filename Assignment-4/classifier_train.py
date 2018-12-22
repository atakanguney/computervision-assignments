import numpy as np
import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from hog import extractHogFromImage, extractHogFromRandomCrop


def extract_features_positive_images(positive_image_paths, block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6)):
    positive_images = [cv.imread(img_path, 0) for img_path in positive_image_paths]
    positive_features = np.vstack([extractHogFromImage(img, block_shape, block_stride, cell_shape) for img in positive_images])
    
    return positive_features

def extract_features_negative_images(negative_image_paths, window_shape=(36, 36), window_stride=(48, 48), block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6)):
    negative_images = [cv.imread(img_path, 0) for img_path in negative_image_paths]
    negative_features = np.concatenate([extractHogFromRandomCrop(img, window_shape, window_stride, block_shape, block_stride, cell_shape) for img in negative_images if img.shape[0] > window_shape[0] and img.shape[1] > window_shape[1]], axis=0)

    return negative_features


def construct_train_test(positive_features, negative_features):
    print(positive_features.shape)
    print(negative_features.shape)
    X = np.concatenate([positive_features, negative_features], axis=0)
    y = np.zeros(X.shape[0])
    y[:positive_features.shape[0]] = 1

    return train_test_split(X, y, test_size=0.2)


def construct_train(positive_features, negative_features):
    X = np.concatenate([positive_features, negative_features], axis=0)
    y = np.zeros(X.shape[0])
    y[:positive_features.shape[0]] = 1

    return X, y


def test_score(y_pred, y_test):
    precision, recall, fscore, support = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-Score: {}".format(fscore))
    print("Support: {}".format(support))
    print("Accuracy: {}".format(accuracy))


def train_classifier(positive_features, negative_features, test=True):

    if test:
        X_train, X_test, y_train, y_test = construct_train_test(positive_features, negative_features)

        classifier = LinearSVC(max_iter=10000)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        test_score(y_pred, y_test)
    else:
        X, y = construct_train(positive_features, negative_features)
        classifier = LinearSVC()
        classifier.fit(X, y)

    return classifier
