import cv2 as cv
import numpy as np
from sklearn.feature_extraction.image import extract_patches, extract_patches_2d


def compute_gradients(img):

    gaussian_blured = cv.GaussianBlur(img, (1, 1), 0, 0)

    kernel_x = np.array([
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1],
        [0],
        [1]
    ])

    gx = cv.filter2D(gaussian_blured, cv.CV_64F, kernel_x)
    gy = cv.filter2D(gaussian_blured, cv.CV_64F, kernel_y)

    return gx, gy


def compute_magnitude_angle(gx, gy):
    gx, gy = np.float64(gx), np.float64(gy)

    magnitude = (gx ** 2 + gy ** 2) ** 0.5
    angle = np.arctan2(gy, gx)  # returns in rads between -pi, +pi
    angle = np.mod(angle, np.pi)
    angle = np.rad2deg(angle)

    return magnitude, angle


def weighted_vote_into_cells(magnitude, angle):
    idx, weight = np.divmod(angle, 20)
    mask_lower = np.equal(idx[..., np.newaxis], np.arange(9))
    mask_upper = np.equal(((idx + 1) % 9)[..., np.newaxis], np.arange(9))

    histogram = mask_lower.astype(np.float64) * (magnitude * (weight / 20))[..., np.newaxis] \
        + mask_upper.astype(np.float64) * (magnitude *
                                           (1 - (weight / 20)))[..., np.newaxis]

    return histogram.sum(axis=(2, 3))


def weighted_vote_into_spatial_and_orientation_cells(magnitude, angle, cell_shape):
    magnitude = extract_patches(
        magnitude, patch_shape=cell_shape, extraction_step=cell_shape)
    angle = extract_patches(angle, patch_shape=cell_shape,
                            extraction_step=cell_shape)

    cell_vectors = weighted_vote_into_cells(magnitude, angle)

    return cell_vectors


def contrast_normalize_over_overlapping_spatial_blocks(cell_vectors, num_cells_in_block, num_cells_in_block_stride):
    patch_shape = (*num_cells_in_block, cell_vectors.shape[-1])
    patch_stride = (*num_cells_in_block_stride, cell_vectors.shape[-1])

    blocks = extract_patches(
        cell_vectors, patch_shape=patch_shape, extraction_step=patch_stride).squeeze()
    norms = ((blocks ** 2).sum(axis=-1)) ** 0.5

    # To be safe with 0 division
    return blocks / (norms + 1e-9)[..., np.newaxis]


def extractHogFromImage(img, block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6)):
    gx, gy = compute_gradients(img)

    magnitude, angle = compute_magnitude_angle(gx, gy)

    cell_vectors = weighted_vote_into_spatial_and_orientation_cells(
        magnitude, angle, cell_shape)

    num_cells_in_block = tuple(np.array(block_shape) // np.array(cell_shape))
    num_cells_in_block_stride = tuple(
        np.array(block_stride) // np.array(cell_shape))

    block_vectors = contrast_normalize_over_overlapping_spatial_blocks(
        cell_vectors, num_cells_in_block, num_cells_in_block_stride)

    return block_vectors.flatten()


def extractHogFromRandomCrop(img, window_shape=(36, 36), window_stride=(48, 48), block_shape=(12, 12), block_stride=(6, 6), cell_shape=(6, 6)):
    patches = extract_patches(img, patch_shape=window_shape, extraction_step=window_stride).reshape(-1, *window_shape)

    hogs = [extractHogFromImage(patch, block_shape, block_stride, cell_shape) for patch in patches]

    return np.array(hogs)
