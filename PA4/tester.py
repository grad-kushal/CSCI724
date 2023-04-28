import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from main import get_apis_with_category_frequency_threshold, preprocess_text


def main():
    data, labels = load_data("clustering")
    number_of_labels = len(labels.unique())
    features = preprocess_features(data)
    features['description'].str.split(' ')

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    # transform dataset into vectors
    X_counts = count_vect.fit_transform(features['description'])
    # transform vectors into weighted vectors
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    # sets K equals to 6
    km = KMeans(n_clusters=number_of_labels, random_state=42)
    s = km.fit(X_tfidf)
    labels = s.labels_
    print(labels)
    # using Silhouette distance to evaluate the clustering
    print(silhouette_score(X_tfidf, labels))

    dbs = DBSCAN(eps=1, min_samples=50)
    s = dbs.fit(X_tfidf)
    labels = s.labels_
    print(labels)
    print(silhouette_score(X_tfidf, labels))


def preprocess_features(features):
    """
    Preprocess the features
    :param features: the features
    :return: the preprocessed features
    """
    features['description'] = features['description'].apply(preprocess_text)
    return features


def load_data(task):
    # Get the data from mongo
    if task == "classification":
        data_from_mongo = get_apis_with_category_frequency_threshold(400)
        print("Data size: ", len(data_from_mongo))
    elif task == "clustering":
        data_from_mongo = get_apis_with_category_frequency_threshold(200)
        print("Data size: ", len(data_from_mongo))

    # Create a dataframe
    df = pd.DataFrame(data_from_mongo)

    # Get the Name, Description, Provider and Tags columns as features
    features = pd.DataFrame(df, columns=['description'])

    return features, df['category']


if __name__ == '__main__':
    main()
