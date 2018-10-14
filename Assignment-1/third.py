import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score

from kmeans import KMeans, calculate_dists
from second import predict_skin_color_masks

# Image path
imgs_base = "data/Images/Original Images/"
n_imgs = 20
imgs = [cv.imread(imgs_base + "img_{0:03d}.jpg".format(i)) for i in range(1, n_imgs + 1)]
# Set seed for reproducibility
seed = 58
np.random.seed(seed=seed)


def normalize(seq, input_start, input_end, output_start, output_end):
    factor = (output_end - output_start) / (input_end - input_start)
    return (output_start + factor * (seq - input_start)).astype(np.uint8)


def color_clustering(img, img_name="image" ,save=True, plot_elbow=False,n_clusters_start=2, n_clusters_end=10, imgs_file="Clustering"):
    kmeans = None
    n_clusters = n_clusters_start
    k = 0
    if not plot_elbow:
        fig = plt.figure(figsize=(30, 30))
    else:
        total_errors = []

    while n_clusters < n_clusters_end + 1:
        # Prepare data for k-Means algorithm
        cvectors = cv.cvtColor(img, cv.COLOR_BGR2RGB).reshape(-1, 3)

        # Create the model
        kmeans = KMeans(n_clusters=n_clusters)
        # Fit the model and get cluster means
        centroids = kmeans.fit(cvectors)
        print(centroids.shape)
        # Get cluster id for each data point in data set
        labels = kmeans.predict(cvectors)
        pred = labels.reshape(img.shape[:-1])

        if plot_elbow:
            errors = calculate_dists(cvectors, centroids)**(1/2)
            mean_errors = []
            for i in range(n_clusters):
                if (labels == i).any():
                    mean_errors.append(errors[labels == i, i].mean())
                else:
                    mean_errors.append(0)

            mean_error = np.mean(mean_errors)

            total_errors.append(mean_error)

        # Initialize img for clusters
        cluster_img = np.zeros(img.shape)
        for i in range(n_clusters):
            cluster_img[np.where(pred == i)] = centroids[i]

        cluster_img = cluster_img.astype(np.uint8)

        if not plot_elbow:
            ax = fig.add_subplot(4, 5, k+1)
            ax.imshow(cluster_img)
            colors = ["Cluster {}".format(i) for i in range(n_clusters)]
            patches = [mpatches.Patch(color=centroids[i] / 255, label=colors[i]) for i in range(len(colors))]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        n_clusters += 1
        k += 1
    if not plot_elbow:
        if save:
            fig.savefig("{}.png".format(imgs_file + "/" + img_name))
        plt.show()

    else:
        plt.plot(total_errors)
        plt.show()
    return kmeans


def cluster_colors(img, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    color_vectors = cv.cvtColor(img, cv.COLOR_BGR2RGB).reshape([-1, 3])
    centroids = kmeans.fit(color_vectors)
    labels = kmeans.predict(color_vectors)
    pred = labels.reshape(img.shape[:-1])

    # Initialize img for clusters
    cluster_img = np.zeros(img.shape)
    for i in range(n_clusters):
        cluster_img[np.where(pred == i)] = centroids[i]

    cluster_img = cluster_img.astype(np.uint8)
    plt.figure(figsize=(10, 10))
    plt.imshow(cluster_img)

    colors = ["Cluster {}".format(i) for i in range(n_clusters)]
    patches = [mpatches.Patch(color=centroids[i] / 255, label=colors[i]) for i in range(len(colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    return kmeans

# Plot clustering results for each image where k from 2 to 10
# for i, img in enumerate(imgs):
#    color_clustering(img, "clusters_{0:03d}".format(i+1), n_clusters_end=20, plot_elbow=True)


# Determined n_clusters for each image
n_clusters = [19, 19, 17, 20, 20, 16, 20, 14, 13, 20, 20, 20, 16, 15, 10, 5, 10, 10, 10, 10]
kmeans_models = []
#for i, img in enumerate(imgs):
#    kmeans_models.append(cluster_colors(img, n_clusters=n_clusters[i]))

cluster_ids = [
    [0, 1, 3, 5, 6],
    [4, 6, 12, 14, 15],
    [4, 13, 14],
    [1, 5, 7, 13, 15, 17],
    [1, 3, 5, 13, 18],
    [3, 5, 6, 13],
    [6, 8, 11, 16],
    [1, 3, 4, 5, 9, 13],
    [1, 2, 3, 8, 10],
    [0, 2, 3, 18],
    [2, 4, 6, 11, 15, 16, 17],
    [5, 10, 15, 16, 19],
    [2, 10, 11],
    [0, 2, 9, 13],
    [4, 5, 6, 8, 9],
    [0, 3],
    [2, 6, 9],
    [0, 1, 2, 7],
    [1, 4, 9],
    [0, 1, 5],
]

def create_binmask(img, kmeans, cluster_ids):
    color_vectors = cv.cvtColor(img, cv.COLOR_BGR2RGB).reshape(-1, 3)
    labels = kmeans.predict(color_vectors).reshape(img.shape[:-1])
    mask = np.zeros(img.shape[:-1]).astype(np.uint8)
    for cluster in cluster_ids:
        mask[labels == cluster] = 255

    return mask


def save_binmask(binmask, fname="mask"):
    plt.imshow(binmask, cmap="gray")
    plt.savefig(fname)

# masks = []

# for i, img in enumerate(imgs):
#    mask = create_binmask(img, kmeans=kmeans_models[i], cluster_ids=cluster_ids[i])
#    save_binmask(mask, "Binary Masks/mask_{0:03d}.png".format(i + 1))
#    masks.append(mask)

# np.save("masks.npy", np.array(masks))


masks = np.load("masks.npy")

skin_color_masks = predict_skin_color_masks(imgs, masks)

for i, color_mask in enumerate(skin_color_masks):
    plt.imshow(color_mask)
    plt.savefig("Skin Color Masks/mask_{0:03d}.png".format(i + 1))
    plt.show()
