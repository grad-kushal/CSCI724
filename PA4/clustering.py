from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_using_kmeans(feature_matrix):
    """
    Cluster using KMeans
    :param feature_matrix: feature matrix
    :return: None
    """
    # Create a K means object
    # kmeans = KMeans(n_clusters=12, random_state=42, metric='cosine')
    kmeans = SpectralClustering(n_clusters=50, random_state=42, assign_labels='kmeans', affinity='polynomial')
    # Fit the training data
    kmeans.fit(feature_matrix)
    # Get cluster labels
    cluster_labels = kmeans.labels_
    # Get cluster centers
    # Evaluate the performance of the model using silhouette score
    ss = silhouette_score(feature_matrix, cluster_labels)
    return ss


def cluster_using_dbscan(feature_matrix, labels):
    """
    Cluster using DBScan
    :param feature_matrix: feature matrix
    :param labels: labels
    :return: None
    """
    eps_values = [0.5, 1]
    # eps_values.reverse()
    min_samples_values = [50, 100]
    min_samples_values.reverse()
    for eps in eps_values:
        for min_samples in min_samples_values:
            print('eps: ', eps, ' min_samples: ', min_samples)
            # Create a DBScan object
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
            # Fit the training data
            dbscan.fit(feature_matrix)
            # Get cluster labels
            cluster_labels = dbscan.labels_
            # Evaluate the performance of the model using silhouette score
            # ss = silhouette_score(feature_matrix, cluster_labels)
            # print('ss: ', ss)
            print('cluster_labels: ', cluster_labels)
    return None
