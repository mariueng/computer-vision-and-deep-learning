import numpy as np
import matplotlib.pyplot as plt

"""
Code originally from: https://github.com/AIZOOTech/object-detection-anchors/blob/master/kmeans.py

Adjusted to match the above codebase.
"""


def iou(boxes, clusters):
    """
    Calculates the Intersection over Union (IoU) between N boxes and K clusters.
    :param boxes: numpy array of shape (n, 2) where n is the number of box, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (n, k) where k is the number of clusters
    """
    N = boxes.shape[0]
    K = clusters.shape[0]
    iw = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 0], (N, K)),    # (N, 1) -> (N, K)
        np.broadcast_to(clusters[np.newaxis, :, 0], (N, K))  # (1, K) -> (N, K)
    )
    ih = np.minimum(
        np.broadcast_to(boxes[:, np.newaxis, 1], (N, K)),
        np.broadcast_to(clusters[np.newaxis, :, 1], (N, K))
    )
    if np.count_nonzero(iw == 0) > 0 or np.count_nonzero(ih == 0) > 0:
        raise ValueError("Some box has no area")

    intersection = iw * ih   # (N, K)
    boxes_area = np.broadcast_to((boxes[:, np.newaxis, 0] * boxes[:, np.newaxis, 1]), (N, K))
    clusters_area = np.broadcast_to((clusters[np.newaxis, :, 0] * clusters[np.newaxis, :, 1]), (N, K))

    iou_ = intersection / (boxes_area + clusters_area - intersection + 1e-7)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean(np.max(iou(boxes, clusters), axis=1))


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    iter_num = 1
    while True:
        # print("Iteration: %d" % iter_num)
        iter_num += 1

        distances = 1 - iou(boxes, clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            if len(boxes[nearest_clusters == cluster]) == 0:
                # print("Cluster %d is zero size" % cluster)
                # to avoid empty cluster
                clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]
                continue

            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

# Utils

def show_cluster(data, cluster, max_points=2000):
    # Display bouding box's size distribution and anchor generated in scatter.
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points)
        data = data[idx]
    plt.scatter(data[:,0], data[:,1], s=5, c='lavender')
    plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Bounding and anchor distribution")
    plt.savefig("cluster.png")
    plt.show()


def show_width_height(data, cluster, bins=50):
    # Display bouding box distribution with histgram.
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    width = data[:, 0]
    height = data[:, 1]
    ratio = height / width

    plt.figure(1,figsize=(20, 6))
    plt.subplot(131)
    plt.hist(width, bins=bins, color='green')
    plt.xlabel('width')
    plt.ylabel('number')
    plt.title('Distribution of Width')

    plt.subplot(132)
    plt.hist(height,bins=bins, color='blue')
    plt.xlabel('Height')
    plt.ylabel('Number')
    plt.title('Distribution of Height')

    plt.subplot(133)
    n = plt.hist(ratio, bins=bins,  color='magenta')
    plt.xlabel('Height / Width')
    plt.ylabel('number')
    plt.title('Distribution of aspect ratio(Height / Width)')
    plt.savefig("shape-distribution.png")
    plt.show()
    return n


def sort_cluster(cluster):
    # Sort the cluster to with area small to big.
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)
    area = cluster[:, 0] * cluster[:, 1]
    cluster = cluster[area.argsort()]
    ratio = cluster[:,1:2] / cluster[:, 0:1]
    return np.concatenate([cluster, ratio], axis=-1)

def sort_cluster_by_histogram_count(cluster, n):
    counts, bin_edges, _ = n
    if cluster.dtype != np.float32:
        cluster = cluster.astype(np.float32)
    aspect_ratios = cluster[:, 1:2] / cluster[:, 0:1]
    ratio = np.concatenate([cluster, aspect_ratios], axis=-1)

    # Change shape of aspect_ratios from (N, 1) to (N, )
    aspect_ratios = aspect_ratios.reshape(-1)

    # Check which bin the ratio is in.
    bin_idx = np.digitize(aspect_ratios, bin_edges)

    # Sort the aspect_ratios by the count of the bin they are in, from big to small.
    counts_per_bin = counts[bin_idx]
    sort_idx = np.argsort(counts_per_bin)[::-1]
    sort_counts = counts_per_bin[sort_idx]

    # Sort the ratio by the sorted index.
    ratio = ratio[sort_idx]

    # Concatenate sort_counts to ratio.
    return np.concatenate([ratio, sort_counts[:, None]], axis=-1)

def run_kmeans(bboxes, n_clusters, bins=50, sort='width', verbose=False):
    out = kmeans(boxes=bboxes, k=n_clusters)

    accuracy = avg_iou(bboxes, out)
    if verbose:
        print("Accuracy: {:.2f}".format(accuracy))

    show_cluster(bboxes, out, max_points=2000)
    n = show_width_height(bboxes, out, bins=bins)

    if sort == 'width':
        out_sorted = sort_cluster(out)
    if sort == 'frequency':
        out_sorted = sort_cluster_by_histogram_count(out, n)

    if out.dtype != np.float32:
        out = out.astype(np.float32)

    if verbose:
        print("Recommended aspect ratios: (width/height)")
        columns = [" width", "height", "ratio", "count"]
        separator = "     "
        if sort == 'width':
            print(separator.join(columns[:2]))
            for i in range(len(out_sorted)):
                print(f"{out_sorted[i][0]:>6.2f}    {out_sorted[i][1]:>6.2f}    {out_sorted[i][2]:>6.2f}")
        else:
            print(separator.join(columns))
            for i in range(len(out_sorted)):
                print(f"{out_sorted[i][0]:>6.2f}    {out_sorted[i][1]:>6.2f}    {out_sorted[i][2]:>6.2f}    {out_sorted[i][3]:>6.2f}")

    return (accuracy, out_sorted)
