from scipy.interpolate import RegularGridInterpolator
import numpy as np
from matplotlib import pyplot as plt


def interest_points(imgs):
    # Get interest points by mouse clicks
    int_points = {}
    for img in imgs:
        fig = plt.figure()
        plt.imshow(imgs[img])
        im_points = plt.ginput(n_interest_points, timeout=-1)
        int_points[img] = im_points
        plt.close(fig)

    return int_points


def computeH(im1Points, im2Points):
    """Computes Homography Matrix with LS

    Parameters
    ----------
    im1Points: ndarray
        List of interest points of image 1
    im2Points: ndarray
        List of interest points of image 2

    Returns
    -------
    ndarray
        Estimated H matrix

    """
    # Make sure points are instances of numpy.ndarray
    if not isinstance(im1Points, np.ndarray):
        im1Points = np.array(im1Points).T

    if not isinstance(im2Points, np.ndarray):
        im2Points = np.array(im2Points).T

    # Normalization transform matrix
    def normalization_transform(img):
        mu1 = img["mu1"]
        mu2 = img["mu2"]

        pow = img["pow"]

        T1 = np.eye(3)
        T2 = np.eye(3)

        # Translation part
        T1[0, 2] = -1 * mu1
        T1[1, 2] = -1 * mu2

        # Scaling part
        T2[0, 0] = 1 / pow
        T2[1, 1] = 1 / pow

        return np.matmul(T2, T1)

    img1 = {
        "mu1": im1Points[0, :].mean(),
        "mu2": im1Points[1, :].mean(),
        "pow": np.linalg.norm(im1Points, axis=1).mean() / np.sqrt(2)
        # "std1": im1Points[0, :].std(),
        # "std2": im1Points[1, :].std()
    }

    img2 = {
        "mu1": im2Points[0, :].mean(),
        "mu2": im2Points[1, :].mean(),
        "pow": np.linalg.norm(im2Points, axis=1).mean() / np.sqrt(2)
        # "std1": im2Points[0, :].std(),
        # "std2": im2Points[1, :].std()
    }

    T1 = normalization_transform(img1)
    T2 = normalization_transform(img2)

    def homogenous(arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        return np.vstack((arr, np.ones(arr.shape[1])))

    normalized_points = {
        "img1": np.matmul(T1, homogenous(im1Points)),
        "img2": np.matmul(T2, homogenous(im2Points))
    }

    def homography_estimation(interested_points):

        img1 = interested_points["img1"]
        img2 = interested_points["img2"]

        n = img1.shape[1]
        d = img1.shape[0]
        A = np.zeros([2 * n, 3 * d])

        odd_idx = [i % 2 == 1 for i in range(2 * n)]
        even_idx = [i % 2 == 0 for i in range(2 * n)]

        A[even_idx, :d] = img1.T
        A[odd_idx, d: 2 * d] = img1.T

        temp_ = np.zeros([2 * n, d])
        temp_[even_idx] = img1.T
        temp_[odd_idx] = img1.T
        A[:, -d:] = -1 * (img2[:-1, :].T.flatten()[:, np.newaxis] * temp_)

        _, _, Vt = np.linalg.svd(A, full_matrices=True)

        return Vt[-1, :].reshape(d, d)

    return np.matmul(np.linalg.pinv(T2), np.matmul(homography_estimation(normalized_points), T1))


def warp(image, H):
    """Warping function with homography matrix

    Warps the source image by using inverse mapping

    Parameters
    ----------
    image: ndarray
        Image to be warped assumed in RGB
    H: ndarray
        Homography matrix

    Returns
    -------
    ndarray
        Warped image

    """

    # Be sure image is in ndarray format
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    h, w, _ = image.shape
    y = np.arange(0, h)
    x = np.arange(0, w)

    interpolation_fun = RegularGridInterpolator((y, x), image, method="nearest")

    Hinv = np.linalg.pinv(H)

    c = np.array([
        [0, 0, w - 1, w - 1],
        [0, h - 1, 0, h - 1],
        [1, 1, 1, 1]
    ])

    cp = np.matmul(H, c)
    cp = cp / (cp[2, :])

    min_ = np.floor(cp.min(axis=1)).astype(np.int)
    max_ = np.ceil(cp.max(axis=1)).astype(np.int)

    xmin = min_[0]
    ymin = min_[1]

    xmax = max_[0]
    ymax = max_[1]

    yp = np.arange(ymin, ymax)
    xp = np.arange(xmin, xmax)

    ypsize = ymax - ymin
    xpsize = xmax - xmin

    mesh = np.meshgrid(xp, yp, indexing="ij")
    new_image = np.zeros([ypsize, xpsize, 3])

    Xp = np.stack([
        mesh[0].flatten(),
        mesh[1].flatten(),
        np.ones(ypsize * xpsize)
    ]).astype(np.int)

    X = np.matmul(Hinv, Xp)
    X = X / (X[2, :][np.newaxis, :])
    X = np.floor(X).astype(np.int)
    y_check = (-1 < X[1, :]) & (X[1, :] < h)
    x_check = (-1 < X[0, :]) & (X[0, :] < w)

    Xp = Xp - Xp.min(axis=1)[:, np.newaxis]

    check = y_check & x_check
    new_image[Xp[1, check], Xp[0, check]] = interpolation_fun(np.stack((X[1, check], X[0, check]), axis=1))

    return new_image, min_[0], max_[0], min_[1], max_[1]


n_interest_points = 15

left1 = plt.imread("cmpe-building/left-1.jpg")
left2 = plt.imread("cmpe-building/left-2.jpg")
mid = plt.imread("cmpe-building/middle.jpg")
right1 = plt.imread("cmpe-building/right-1.jpg")
right2 = plt.imread("cmpe-building/right-2.jpg")

imgs = {
    "left2": left2,
    "left1": left1,
    "mid": mid,
    "right1": right1,
    "right2": right2
}

img_names = list(imgs.keys())

img_pairs = list(zip(img_names[:-1], img_names[1:]))

imgs = {
    "left1": left1,
    "mid": mid
}
int_points = interest_points(imgs)
H1 = computeH(int_points["left1"], int_points["mid"])
warped1, xmin1, xmax1, ymin1, ymax1 = warp(imgs["left1"], H1)

imgs = {
    "left2": left2,
    "mid": mid
}
int_points = interest_points(imgs)
H2 = computeH(int_points["left2"], int_points["mid"])
warped2, xmin2, xmax2, ymin2, ymax2 = warp(imgs["left2"], H2)

imgs = {
    "right1": right1,
    "mid": mid
}
int_points = interest_points(imgs)
H3 = computeH(int_points["right1"], int_points["mid"])
warped3, xmin3, xmax3, ymin3, ymax3 = warp(imgs["right1"], H3)

imgs = {
    "right2": right2,
    "mid": mid
}
int_points = interest_points(imgs)
H4 = computeH(int_points["right2"], int_points["mid"])
warped4, xmin4, xmax4, ymin4, ymax4 = warp(imgs["right2"], H4)

xmin = np.min([xmin1, xmin2, 0, xmin3, xmin4])
xmax = np.max([xmax1, xmax2, imgs["mid"].shape[1], xmax3, xmax4])

ymin = np.min([ymin1, ymin2, 0, ymin3, ymin4])
ymax = np.max([ymax1, ymax2, imgs["mid"].shape[0], ymax3, ymax4])

xsize = xmax - xmin
ysize = ymax - ymin

merged = np.zeros([ysize, xsize, 3])

merged[ymin2 - ymin: ymin2 - ymin + warped2.shape[0], :warped2.shape[1]] = warped2

mask1 = warped1.sum(axis=2).nonzero()
merged[ymin1 - ymin: ymin1 - ymin + warped1.shape[0], xmin1 - xmin: xmin1 - xmin + warped1.shape[1]][mask1] = warped1[mask1]

mask4 = warped4.sum(axis=2).nonzero()
merged[ymin4 - ymin: ymin4 - ymin + warped4.shape[0], xmin4 - xmin: xmin4 - xmin + warped4.shape[1]][mask4] = warped4[mask4]

mask3 = warped3.sum(axis=2).nonzero()
merged[ymin3 - ymin: ymin3 - ymin + warped3.shape[0], xmin3 - xmin: xmin3 - xmin + warped3.shape[1]][mask3] = warped3[mask3]

merged[-ymin: -ymin + imgs["mid"].shape[0], -xmin: -xmin + imgs["mid"].shape[1]] = np.maximum(
    merged[-ymin: -ymin + imgs["mid"].shape[0], -xmin: -xmin + imgs["mid"].shape[1]],
    imgs["mid"]
)

merged = merged.astype(np.uint8)
plt.imshow(merged)
plt.show()
