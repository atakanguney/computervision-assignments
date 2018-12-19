import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches


def compute_gradients(img, ksize=3):
    gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)

    magnitudes = (gx ** 2 + gy ** 2) ** 0.5

    angles = np.rad2deg(np.arctan(gy, gx))
    angles[angles < 0] = angles[angles < 0] + 180.0

    return magnitudes, angles


def divide_cells(blocks_magnitudes, blocks_angles, cell_size):
    n_rows, n_cols = blocks_magnitudes.shape[:2]
    cells_magnitudes = []
    cells_angles = []
    idx = np.ndindex((n_rows, n_cols))
    for ii in idx:
        cells_magnitudes.append(
            extract_patches(blocks_magnitudes[ii], patch_shape=cell_size, extraction_step=cell_size))
        cells_angles.append(extract_patches(blocks_angles[ii], patch_shape=cell_size, extraction_step=cell_size))

    n_rows_c, n_cols_c = cells_magnitudes[0].shape[:2]
    cells_magnitudes = np.array(cells_magnitudes).reshape([n_rows, n_cols, n_rows_c, n_cols_c, -1])
    cells_angles = np.array(cells_angles).reshape([n_rows, n_cols, n_rows_c, n_cols_c, -1])

    return cells_magnitudes, cells_angles


def divide_blocks(magnitude, angle, block_size, block_stride):
    blocks_magnitudes = extract_patches(magnitude, patch_shape=block_size, extraction_step=block_stride)
    blocks_angles = extract_patches(angle, patch_shape=block_size, extraction_step=block_stride)

    return blocks_magnitudes, blocks_angles


def _weighted_hist(angle, magnitude):
    idx = np.int64(angle // 20)

    prop1 = angle - idx * 20.0
    prop2 = 20.0 - prop1

    lambda1 = prop2 / 20.0
    lambda2 = prop1 / 20.0

    idx1 = idx % 9
    idx2 = (idx + 1) % 9

    mag1 = lambda1 * magnitude
    mag2 = lambda2 * magnitude

    hist = []
    for i in range(9):
        hist.append(mag1[idx1 == i].sum() + mag2[idx2 == i].sum())

    return hist


def weighted_hist_vectors(cells_magnitudes, cells_angles):
    n_blocks_row, n_blocks_col, n_cells_row, n_cells_col = cells_magnitudes.shape[:4]

    idx = np.ndindex((n_blocks_row, n_blocks_col, n_cells_row, n_cells_col))
    weighted_hists = []
    for ii in idx:
        weighted_hists.append(_weighted_hist(cells_angles[ii], cells_magnitudes[ii]))

    return np.array(weighted_hists).reshape([n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, -1])


def group_and_normalize(hist_vectors):
    n_blocks_row, n_blocks_col = hist_vectors.shape[:2]

    grouped = hist_vectors.reshape([n_blocks_row, n_blocks_col, -1])
    normalized = np.apply_along_axis(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x, axis=2,
                                     arr=grouped)

    return normalized


def hog_features(img, block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):
    # Compute gradients
    magnitude, angle = compute_gradients(img)

    # Blocks
    blocks_magnitudes, blocks_angles = divide_blocks(magnitude, angle, block_size, block_stride)

    # Cells
    cells_magnitudes, cells_angles = divide_cells(blocks_magnitudes, blocks_angles, cell_size)

    # Compute weighted histogram vectors
    hist_vectors = weighted_hist_vectors(cells_magnitudes, cells_angles)

    # Group and normalize
    normalized_hists = group_and_normalize(hist_vectors)

    return normalized_hists.flatten()


def extractHogFromImage(img_path, block_size=(12, 12), block_stride=(6, 6), cell_size=(6, 6)):
    img = cv.imread(img_path, 0)
    return hog_features(img, block_size, block_stride, cell_size)


def extractFromRandomCrop(img_path, win_size=(36, 36), win_stride=(48, 48), block_size=(12, 12), block_stride=(6, 6),
                          cell_size=(6, 6)):
    img = cv.imread(img_path, 0)

    hogs = []
    patches = extract_patches(img, patch_shape=win_size, extraction_step=win_stride)
    for patch in np.ndindex(patches.shape[:2]):
        hogs.append(hog_features(patches[patch], block_size=block_size, block_stride=block_stride, cell_size=cell_size))

    return hogs
