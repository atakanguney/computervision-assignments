import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches, extract_patches_2d


def compute_gradients(img, ksize=3):
    gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
    gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)

    magnitudes = (gx ** 2 + gy ** 2) ** 0.5

    angles = np.rad2deg(np.arctan(gy, gx))
    angles[angles < 0] = angles[angles < 0] + 180.0

    return magnitudes, angles


def _cell_indexes(img_shape, cell_size):
    n_rows, n_cols = img_shape
    c_rows, c_cols = cell_size

    assert n_rows > c_rows and n_cols > c_cols

    cells_r = n_rows // c_rows
    cells_c = n_cols // c_cols

    row, col = np.indices(cell_size)

    for r in range(cells_r):
        for c in range(cells_c):
            yield row + r * c_rows, col + c * c_rows


def divide_cells(magnitudes, angles, cell_size):
    img_shape = magnitudes.shape

    for row, col in _cell_indexes(img_shape, cell_size):
        yield magnitudes[row, col], angles[row, col]



def divide_blocks(magnitude, angle, block_size, block_stride):
    blocks_magnitudes = extract_patches(magnitude, patch_shape=block_size, extraction_step=block_stride)
    blocks_angles = extract_patches(angle, patch_shape=block_size, extraction_step=block_stride)

    return blocks_magnitudes, blocks_angles


def _weighted_hist(matrix):
    half = len(matrix) // 2
    magnitude = matrix[:half]
    angle = matrix[half:]

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

    return np.array(hist)


def weighted_hist_vectors(magnitudes, angles, cell_size):
    img_shape = magnitudes.shape

    n_rows, n_cols = img_shape
    c_rows, c_cols = cell_size

    assert n_rows > c_rows and n_cols > c_cols

    cells_r = n_rows // c_rows
    cells_c = n_cols // c_cols

    cells_magnitudes = []
    cells_angles = []
    magang = []
    for cell_mag, cell_ang in divide_cells(magnitudes, angles, cell_size):
        cells_magnitudes.append(cell_mag)
        cells_angles.append(cell_ang)
        magang.append(np.concatenate([cell_mag.flatten(), cell_ang.flatten()]))

    magang = np.array(magang)
    hists = np.apply_along_axis(_weighted_hist, axis=1, arr=magang)
    return hists.reshape([cells_r, cells_c, -1])


def _group_and_normalize(hist_vectors, block_size, block_stride):
    # TODO: Make use of block stride
    blocks = extract_patches_2d(hist_vectors, patch_size=block_size)
    flattened = blocks.reshape([blocks.shape[0], -1])
    normalized = np.apply_along_axis(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x, axis=1,
                                     arr=flattened)

    return normalized


def hog_features(img, block_size=(2, 2), block_stride=(1, 1), cell_size=(6, 6)):

    magnitude, angle = compute_gradients(img)

    hist_vectors = weighted_hist_vectors(magnitude, angle, cell_size)

    normalized = _group_and_normalize(hist_vectors, block_size, block_stride)

    return normalized.flatten()


def extractHogFromImage(img_path, block_size=(2, 2), block_stride=(1, 1), cell_size=(6, 6)):
    img = cv.imread(img_path, 0)
    return hog_features(img, block_size, block_stride, cell_size)


def extractFromRandomCrop(img_path, win_size=(36, 36), win_stride=(48, 48), block_size=(2, 2), block_stride=(1, 1),
                          cell_size=(6, 6)):
    img = cv.imread(img_path, 0)

    hogs = []
    patches = extract_patches(img, patch_shape=win_size, extraction_step=win_stride)
    for patch in np.ndindex(patches.shape[:2]):
        hogs.append(
            hog_features(patches[patch], block_size=block_size, block_stride=block_stride, cell_size=cell_size))

    return hogs
