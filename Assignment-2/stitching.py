from scipy.interpolate import RegularGridInterpolator
import numpy as np
from matplotlib import pyplot as plt


def interest_points(imgs, n_interest_points=15):
    # Get interest points by mouse clicks
    int_points = {}
    for img in imgs:
        fig = plt.figure()
        plt.imshow(imgs[img])
        plt.title("Select {} points".format(n_interest_points))
        im_points = plt.ginput(n_interest_points, timeout=-1)
        int_points[img] = im_points
        plt.close(fig)

    return int_points


def computeH(im1Points, im2Points, normalized):
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
        im1Points = np.array(im1Points)

    if not isinstance(im2Points, np.ndarray):
        im2Points = np.array(im2Points)

    im1Points = im1Points.T
    im2Points = im2Points.T

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
    }

    img2 = {
        "mu1": im2Points[0, :].mean(),
        "mu2": im2Points[1, :].mean(),
        "pow": np.linalg.norm(im2Points, axis=1).mean() / np.sqrt(2)
    }

    T1 = normalization_transform(img1)
    T2 = normalization_transform(img2)

    def homogenous(arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        return np.vstack((arr, np.ones(arr.shape[1])))

    points = {
        "img1": homogenous(im1Points),
        "img2": homogenous(im2Points)
    }

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

    if normalized:
        return np.matmul(np.linalg.pinv(T2), np.matmul(homography_estimation(normalized_points), T1))
    else:
        return homography_estimation(points)


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


def save_points(fname, points):
    with open(fname, "a") as file:
        for img in points:
            file.write("Points for image {} \n".format(img))
            file.write("x y \n")
            for point in points[img]:
                file.write("{} {} \n".format(point[0], point[1]))


def add_noise(variance, points):
    new_points = {}
    for img in points:
        new_points[img] = np.array(points[img]) + np.random.rand(len(points[img]), 2) * variance
    return new_points


def merge_image(merged, img, xminim, yminim, xmin, ymin, method="add"):
    return merged


def merge_images(imgs, xmaxs, xmins, ymaxs, ymins, method="add"):
    xmin = np.min(xmins)
    xmax = np.max(xmaxs)

    ymin = np.min(ymins)
    ymax = np.max(ymaxs)

    xsize = xmax - xmin
    ysize = ymax - ymin
    print(xsize)
    merged = np.zeros([ysize, xsize, 3])

    for i, img in enumerate(imgs):
        mask = (img.sum(axis=2) > 0)
        mask_ = (merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0],
                 xmins[i] - xmin: xmins[i] - xmin + img.shape[1]].sum(
            axis=2) > 0)
        mask_1 = mask & mask_
        mask_2 = mask & (~mask_)

        if method == "add":
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_1] = img[mask_1]
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_2] = img[mask_2]
        elif method == "max":
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_1] = \
                np.maximum(img[mask_1],
                           merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0],
                           xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][mask_1])
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_2] = img[mask_2]
        elif method == "avg":
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_1] = \
                (img[mask_1] +
                 merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0],
                 xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][mask_1]) / 2
            merged[ymins[i] - ymin: ymins[i] - ymin + img.shape[0], xmins[i] - xmin: xmins[i] - xmin + img.shape[1]][
                mask_2] = img[mask_2]

    return merged.astype(np.uint8)


def get_warpeds(imgs, fname, normalized=True, variance=1, n_interest_points=15, noise=False):
    warpeds = []
    xmaxs = []
    xmins = []
    ymaxs = []
    ymins = []

    for img_pair in imgs:
        imgs_ = {
            "first": img_pair[0],
            "second": img_pair[1]
        }
        int_points = interest_points(imgs_, n_interest_points)
        save_points(fname, int_points)
        if noise:
            int_points = add_noise(variance, int_points)

        H = computeH(int_points["first"], int_points["second"], normalized)
        warped, xmin, xmax, ymin, ymax = warp(imgs_["first"], H)
        warpeds.append(warped)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        xmins.append(xmin)
        ymins.append(ymin)

    return warpeds, xmaxs, xmins, ymaxs, ymins


def experiment(imgs, ref_img, fname, normalized=True, variance=1, method="add", n_interest_points=15, noise=False):
    imgs_ = []
    for img in imgs:
        imgs_.append((img, ref_img))

    warpeds, xmaxs, xmins, ymaxs, ymins = get_warpeds(imgs_, fname, normalized, variance, n_interest_points, noise)

    warpeds.append(ref_img)
    xmaxs.append(ref_img.shape[1])
    ymaxs.append(ref_img.shape[0])
    xmins.append(0)
    ymins.append(0)

    return merge_images(warpeds, xmaxs, xmins, ymaxs, ymins, method)


if __name__ == "__main__":
    left1 = plt.imread("cmpe-building/left-1.jpg")
    left2 = plt.imread("cmpe-building/left-2.jpg")
    mid = plt.imread("cmpe-building/middle.jpg")
    right1 = plt.imread("cmpe-building/right-1.jpg")
    right2 = plt.imread("cmpe-building/right-2.jpg")

    imgs = [left1, right1]
    ref_img = mid


    def exp_results(exp_name, imgs, ref_img, normalized=True, variance=1, method="add", n_interest_points=15,
                    noise=False):
        points_file = exp_name + "-points.txt"
        merged = experiment(imgs, ref_img, points_file, normalized, variance, method, n_interest_points, noise=noise)
        plt.imshow(merged)
        plt.savefig(exp_name + ".png")
        plt.title(exp_name)
        plt.show()


    exp_name = "5-correspondence"
    exp_results(exp_name, imgs, ref_img, n_interest_points=5)
    exp_name = "12-correspondence"
    exp_results(exp_name, imgs, ref_img, n_interest_points=12)
    exp_name = "12-3-wrong-unnormalized"
    exp_results(exp_name, imgs, ref_img, n_interest_points=12)
    exp_name = "12-3-wrong-normalized"
    exp_results(exp_name, imgs, ref_img, n_interest_points=12)
    exp_name = "12-5-wrong-normalized"
    exp_results(exp_name, imgs, ref_img, n_interest_points=12)

    variances = [1, 5, 10]

    for var in variances:
        exp_name = "gaussian-noise-{}".format(var)
        exp_results(exp_name, imgs, ref_img, n_interest_points=12, variance=var, noise=True)

    variances = [1, 5]

    for var in variances:
        exp_name = "gaussian-noise-unnormalized-{}".format(var)
        exp_results(exp_name, imgs, ref_img, normalized=False, n_interest_points=12, variance=var, noise=True)

    imgs = [left2, left1, right1, right2]
    options = ["avg", "max"]

    for option in options:
        exp_name = "all-" + option
        exp_results(exp_name, imgs, ref_img, n_interest_points=15, method=option)
