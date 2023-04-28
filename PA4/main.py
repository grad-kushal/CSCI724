import string

import visualize as visualize
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.stem import PorterStemmer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import clustering
import database
from sklearn.model_selection import train_test_split
import pandas as pd

import warnings

import classifiers
import feature_extraction
import modeling
import pyfiglet

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.options.mode.chained_assignment = None

# Download stop words list
nltk.download('stopwords')

# Download Porter stemmer
nltk.download('punkt')
ps = PorterStemmer()


def preprocess_text(text):
    """
    Preprocess the text
    :param text: text
    :return: preprocessed text
    """
    text = text.lower()
    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    tokens = [ps.stem(token) for token in tokens]
    # Join the tokens
    text = ' '.join(tokens)
    return text


def split_dataset(features, labels):
    """
    Split the dataset into training and testing sets
    :param features: features
    :param labels: labels
    :return: the training and testing sets
    """
    return train_test_split(features, labels, test_size=0.2, random_state=0)


def perform_clustering(tfidf_feature_matrix, word_embeddings_feature_matrix, lda_feature_matrix, labels):
    """
    Perform clustering
    :param lda_feature_matrix: Feature matrix using LDA
    :param word_embeddings_feature_matrix: Feature matrix using word embeddings
    :param tfidf_feature_matrix:  Feature matrix using TF-IDF
    :param labels: labels
    :return: None
    """
    # Perform clustering using K-Means
    print("K-Means")
    ss = clustering.cluster_using_kmeans(tfidf_feature_matrix)
    print('\tTF-IDF Silhouette score: \t\t', ss)
    ss = clustering.cluster_using_kmeans(word_embeddings_feature_matrix)
    print('\tWE Silhouette score: \t\t\t', ss)
    ss = clustering.cluster_using_kmeans(lda_feature_matrix)
    print('\tLDA Silhouette score: \t\t\t', ss)

    # Perform clustering using DBSCAN
    print("DBSCAN")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(tfidf_feature_matrix.toarray())
    ss = clustering.cluster_using_dbscan(X_std, labels)
    print('\tTF-IDF Silhouette score: \t\t', ss)
    ss = clustering.cluster_using_dbscan(word_embeddings_feature_matrix, labels)
    print('\tWE Silhouette score: \t\t\t', ss)
    ss = clustering.cluster_using_dbscan(lda_feature_matrix, labels)
    print('\tLDA Silhouette score: \t\t\t', ss)


def perform_classification(tfidf_feature_matrix, word_embeddings_feature_matrix, lda_feature_matrix, labels):
    """
    Perform classification
    :param tfidf_feature_matrix: feature matrix using tfidf
    :param word_embeddings_feature_matrix: feature matrix using word embeddings
    :param lda_feature_matrix: feature matrix using LDA
    :param labels: labels
    :return: None
    """
    # Split the dataset into training and testing sets for tfidf
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = split_dataset(tfidf_feature_matrix, labels)
    # Split the dataset into training and testing sets for lda
    X_train_lda, X_test_lda, y_train_lda, y_test_lda = split_dataset(lda_feature_matrix, labels)
    # Split the dataset into training and testing sets for word embeddings
    X_train_we, X_test_we, y_train_we, y_test_we = split_dataset(word_embeddings_feature_matrix, labels)

    # Classify the using Naive Bayes
    print("Naive Bayes")
    y_pred_nb = classifiers.classify_using_multinomial_naive_bayes(X_train_tfidf, X_test_tfidf, y_train_tfidf)
    print("\tTF-IDF Accuracy: \t\t", accuracy_score(y_test_tfidf, y_pred_nb) * 100, "%")
    y_pred_nb = classifiers.classify_using_gaussian_naive_bayes(X_train_we, X_test_we, y_train_we)
    print("\tWE Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_nb) * 100, "%")
    y_pred_nb = classifiers.classify_using_multinomial_naive_bayes(X_train_lda, X_test_lda, y_train_lda)
    print("\tLDA Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_nb) * 100, "%")

    # Classify the using KNN
    print("KNN")
    y_pred_knn = classifiers.classify_using_knn(X_train_tfidf, X_test_tfidf, y_train_tfidf, 200)
    print("\tTF-IDF Accuracy: \t\t", accuracy_score(y_test_tfidf, y_pred_knn) * 100, "%")
    y_pred_knn = classifiers.classify_using_knn(X_train_we, X_test_we, y_train_we, 200)
    print("\tWE Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_knn) * 100, "%")
    y_pred_knn = classifiers.classify_using_knn(X_train_lda, X_test_lda, y_train_lda, 200)
    print("\tLDA Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_knn) * 100, "%")

    # Classify the using SVM
    print("SVM")
    y_pred_svm = classifiers.classify_using_svm(X_train_tfidf, X_test_tfidf, y_train_tfidf)
    print("\tTF-IDF Accuracy: \t\t", accuracy_score(y_test_tfidf, y_pred_svm) * 100, "%")
    y_pred_svm = classifiers.classify_using_svm(X_train_we, X_test_we, y_train_we)
    print("\tWE Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_svm) * 100, "%")
    y_pred_svm = classifiers.classify_using_svm(X_train_lda, X_test_lda, y_train_lda)
    print("\tLDA Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_svm) * 100, "%")

    # Classify using decision tree
    print("Decision Tree")
    y_pred_dt = classifiers.classify_using_decision_tree(X_train_tfidf, X_test_tfidf, y_train_tfidf)
    print("\tTF-IDF Accuracy: \t\t", accuracy_score(y_test_tfidf, y_pred_dt) * 100, "%")
    y_pred_dt = classifiers.classify_using_decision_tree(X_train_we, X_test_we, y_train_we)
    print("\tWE Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_dt) * 100, "%")
    y_pred_dt = classifiers.classify_using_decision_tree(X_train_lda, X_test_lda, y_train_lda)
    print("\tLDA Accuracy: \t\t\t", accuracy_score(y_test_tfidf, y_pred_dt) * 100, "%")


def get_apis_with_category_frequency_threshold(threshold):
    """
    Get the apis with category frequency threshold
    :param threshold: the frequency threshold
    :return: the apis
    """
    # Initialize the database
    mydb, api_data = database.inti_database()
    # Get category frequencies
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$lookup": {
            "from": "apis",
            "localField": "_id",
            "foreignField": "category",
            "as": "apis"
        }},
        {"$match": {"count": {"$gte": threshold}}}
    ]
    print("Getting data from mongoDB")
    categories_with_frequencies = database.execute_aggregation(mydb, 'apis', pipeline)
    api_ids = []
    count = 0
    for category in categories_with_frequencies:
        count += 1
        for api in category['apis']:
            api_ids.append(api['_id'])
    print("Total categories: ", count)
    selected_apis = database.get_documents(mydb, 'apis', 'all', 'all', 'all', 'all', 'all', 'all', api_ids)
    data = []
    for api in selected_apis:
        data.append(api)

    database.close_database(mydb)
    return data


def preprocess_features(features):
    """
    Preprocess the features
    :param features: the features
    :return: the preprocessed features
    """
    features['name'] = features['name'].apply(preprocess_text)
    features['description'] = features['description'].apply(preprocess_text)
    features['provider'] = features['provider'].apply(preprocess_text)
    features['tags'] = features['tags'].apply(preprocess_text)
    features['summary'] = features['summary'].apply(preprocess_text)
    return features


def get_features_for(task):
    # Get the data from mongo
    if task == "classification":
        data_from_mongo = get_apis_with_category_frequency_threshold(400)
        print("Data size: ", len(data_from_mongo))
    elif task == "clustering":
        data_from_mongo = get_apis_with_category_frequency_threshold(50)
        print("Data size: ", len(data_from_mongo))

    # Create a dataframe
    df = pd.DataFrame(data_from_mongo)

    # Get the Name, Description, Provider and Tags columns as features
    features = df[['name', 'description', 'provider', 'tags', 'summary']]
    features['tags'] = features['tags'].apply(lambda x: ' '.join(x))

    return features, df['category']


def visualize_data(X_tf_idf, X_word2vec, X_lda, labels):
    """
    Visualize the data
    :param X_tf_idf: the tf-idf feature matrix
    :param X_word2vec: the word2vec feature matrix
    :param X_lda: the LDA feature matrix
    :param labels: the labels
    :return: None
    """
    # define the number of components for the SVD
    n_components = 2

    # create the pipeline to transform the data with TruncatedSVD
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    # fit and transform the data with the pipeline
    X_lsa = lsa.fit_transform(X_tf_idf)

    # plot the data
    plt.scatter(X_lsa[:, 0], X_lsa[:, 1])
    plt.show()
    plt.savefig('plot.png')


def main():
    """
    Main function
    :return: None
    """
    ascii_banner = pyfiglet.figlet_format("CLASSIFICATION")
    print('\n\n' + ascii_banner)
    features, labels = get_features_for("classification")
    features = preprocess_features(features)
    tfidf_feature_matrix, word_embeddings_feature_matrix, lda_feature_matrix = modeling.get_feature_matrices(features)
    perform_classification(tfidf_feature_matrix, word_embeddings_feature_matrix, lda_feature_matrix, labels)

    ascii_banner = pyfiglet.figlet_format("CLUSTERING")
    print('\n\n' + ascii_banner)
    features, labels = get_features_for("clustering")
    features = preprocess_features(features)
    X_tf_idf, X_word2vec, X_lda = feature_extraction.extract_features(features)
    # visualize_data(X_tf_idf, X_word2vec, X_lda, labels)
    perform_clustering(X_tf_idf, X_word2vec, X_lda, labels)


if __name__ == '__main__':
    main()
