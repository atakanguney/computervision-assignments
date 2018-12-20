import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches, extract_patches_2d


def compute_gradients(img, ksize=3):
    gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)

    magnitudes = ((gx ** 2 + gy ** 2) ** 0.5)

    angles = np.arctan2(gy, gx)
    angles[angles < 0] = angles[angles < 0] + np.pi
    angles = np.rad2deg(angles)

    return magnitudes, angles


def _divide_cells(block_magnitudes, block_angles, cell_size):
    w_rows, w_cols, b_rows, b_cols, b_height, b_width = block_magnitudes.shape
    cell_height, cell_width = cell_size

    mag_strides = block_magnitudes.strides
    ang_strides = block_angles.strides

    cell_rows = b_height // cell_height
    cell_cols = b_width // cell_width

    new_shape = (
        w_rows, w_cols,
        b_rows, b_cols,
        cell_rows, cell_cols,
        cell_height, cell_width,
    )

    new_mag_strides = (
        mag_strides[0],
        mag_strides[1],
        mag_strides[2],
        mag_strides[3],
        mag_strides[4] * cell_height,
        mag_strides[5] * cell_width,
        mag_strides[4],
        mag_strides[5],
    )

    new_ang_strides = (
        ang_strides[0],
        ang_strides[1],
        ang_strides[2],
        ang_strides[3],
        ang_strides[4] * cell_height,
        ang_strides[5] * cell_width,
        ang_strides[4],
        ang_strides[5],
    )

    cell_magnitudes = np.lib.stride_tricks.as_strided(
        block_magnitudes,
        shape=new_shape,
        strides=new_mag_strides
    )

    cell_angles = np.lib.stride_tricks.as_strided(
        block_angles,
        shape=new_shape,
        strides=new_ang_strides
    )

    return cell_magnitudes, cell_angles


def _divide_blocks(window_magnitude, window_angle, block_size, block_stride):

    w_rows, w_cols, w_height, w_width = window_magnitude.shape

    b_height, b_width = block_size
    bs_height, bs_width = block_stride

    mag_strides = window_magnitude.strides
    ang_strides = window_angle.strides

    b_rows = (w_height - b_height) // bs_height + 1
    b_cols = (w_width - b_width) // bs_width + 1

    new_shape = (
        w_rows, w_cols,
        b_rows, b_cols,
        b_height, b_width,
    )

    new_mag_strides = (
        mag_strides[0],
        mag_strides[1],
        mag_strides[2] * bs_height,
        mag_strides[3] * bs_width,
        mag_strides[2],
        mag_strides[3],
    )

    new_ang_strides = (
        ang_strides[0],
        ang_strides[1],
        ang_strides[2] * bs_height,
        ang_strides[3] * bs_width,
        ang_strides[2],
        ang_strides[3],
    )

    block_magnitudes = np.lib.stride_tricks.as_strided(
        window_magnitude,
        shape=new_shape,
        strides=new_mag_strides
    )

    block_angles = np.lib.stride_tricks.as_strided(
        window_angle,
        shape=new_shape,
        strides=new_ang_strides
    )

    return block_magnitudes, block_angles


def _divide_windows(magnitude, angle, window_size, window_stride):
    window_magnitudes = extract_patches(
        magnitude, patch_shape=window_size, extraction_step=window_stride)
    window_angles = extract_patches(
        angle, patch_shape=window_size, extraction_step=window_stride)

    return window_magnitudes, window_angles


def _weighted_hist(magnitudes, angles):

    idx = np.int64(angles // 20)

    prop1 = angles - idx * 20.0
    prop2 = 20.0 - prop1

    lambda1 = prop2 / 20.0
    lambda2 = prop1 / 20.0

    idx1 = idx % 9
    idx2 = (idx + 1) % 9

    mag1 = lambda1 * magnitudes
    mag2 = lambda2 * magnitudes

    hists = []

    for i in range(9):
        hists.append((mag1 * (idx1 == i)).sum(axis=(-1, -2)) +
                     (mag2 * (idx2 == i)).sum(axis=(-1, -2)))

    hists = np.array(hists)

    return np.moveaxis(hists, 0, -1)


def weighted_hist_vectors(magnitudes, angles, block_stride, cell_size):

    cell_magnitudes, cell_angles = _divide_cells(magnitudes, angles, cell_size)

    hists = _weighted_hist(cell_magnitudes, cell_angles)

    return hists


def group_and_normalize(hist_vectors):
    w_rows, w_cols, b_rows, b_cols, cell_rows, cell_cols, nbins = hist_vectors.shape

    flattened = hist_vectors.reshape(
        [w_rows * w_cols * b_rows * b_cols, cell_rows * cell_cols * nbins])

    flattened = np.float64(flattened)
    norms = np.sqrt(np.einsum("...i,...i", flattened, flattened))
    epsilon = 10 ** -8
    normalized = flattened / (norms + epsilon)[:, np.newaxis]

    return normalized


def hog_features(img, window_size=(36, 36), window_stride=(48, 48), block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):

    magnitude, angle = compute_gradients(img)

    window_magnitudes, window_angles = _divide_windows(
        magnitude, angle, window_size, window_stride)

    block_magnitudes, block_angles = _divide_blocks(
        window_magnitudes, window_angles, block_size, block_stride)

    cell_magnitudes, cell_angles = _divide_cells(
        block_magnitudes, block_angles, cell_size)

    hist_vectors = _weighted_hist(cell_magnitudes, cell_angles)

    normalized = group_and_normalize(hist_vectors)

    return normalized.flatten()


def extractHogFromImage(img_path, block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):
    img = cv.imread(img_path, 0)
    img = np.float64(img) / 255.0
    return hog_features(img, block_size=block_size, block_stride=block_stride, cell_size=cell_size)


def extractHogFromRandomCrop(img_path, win_size=(36, 36), win_stride=(48, 48), block_size=(12, 12), block_stride=(6, 6),
                             cell_size=(6, 6)):
    img = cv.imread(img_path, 0)
    img = np.float64(img) / 255.0

    hogs = hog_features(img, win_size, win_stride,
                        block_size, block_stride, cell_size)
    return np.array(hogs)
