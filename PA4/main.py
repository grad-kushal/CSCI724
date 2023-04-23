import string

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.stem import PorterStemmer

import database
from sklearn.model_selection import train_test_split
import pandas as pd

import warnings

import classifiers
import topic_modeling

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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


def main():
    """
    Main function
    :return: None
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
        {"$match": {"count": {"$gte": 400}}}
    ]
    categories_with_frequencies = database.execute_aggregation(mydb, 'apis', pipeline)
    api_ids = []
    for category in categories_with_frequencies:
        for api in category['apis']:
            api_ids.append(api['_id'])

    selected_apis = database.get_documents(mydb, 'apis', 'all', 'all', 'all', 'all', 'all', 'all', api_ids)
    data = []
    for api in selected_apis:
        data.append(api)

    # Create a dataframe
    df = pd.DataFrame(data)
    # Get the Name, Description, Provider and Tags columns as features
    features = df[['name', 'description', 'provider', 'tags', 'summary']]
    features['tags'] = features['tags'].apply(lambda x: ' '.join(x))
    # Preprocess the features
    features['name'] = features['name'].apply(preprocess_text)
    features['description'] = features['description'].apply(preprocess_text)
    features['provider'] = features['provider'].apply(preprocess_text)
    features['tags'] = features['tags'].apply(preprocess_text)
    features['summary'] = features['summary'].apply(preprocess_text)
    # Get the category column as labels
    labels = df['category']

    # Apply different topic modeling techniques
    tfidf_feature_matrix = topic_modeling.apply_tfidf_vectorizer(features)
    word_embeddings_feature_matrix = topic_modeling.apply_word_embeddings(features)
    # topic_modeling_feature_matrix = topic_modeling.apply_topic_modeling(features)

    # Split the dataset into training and testing sets for tfidf
    X_train, X_test, y_train, y_test = split_dataset(tfidf_feature_matrix, labels)
    # Classify the using Naive Bayes
    y_pred_nb = classifiers.classify_multinomial_naive_bayes(X_train, X_test, y_train)
    # Classify the using KNN
    y_pred_knn = classifiers.classify_knn(X_train, X_test, y_train, 200)
    # Classify the using SVM
    y_pred_svm = classifiers.classify_svm(X_train, X_test, y_train)
    # Classify using decision tree
    y_pred_dt = classifiers.classify_decision_tree(X_train, X_test, y_train)
    # Print the accuracy scores
    print("TF-IDF")
    print("\tAccuracy score for MNB: ", accuracy_score(y_test, y_pred_nb) * 100, "%")
    print("\tAccuracy score for KNN: ", accuracy_score(y_test, y_pred_knn) * 100, "%")
    print("\tAccuracy score for SVM: ", accuracy_score(y_test, y_pred_svm) * 100, "%")
    print("\tAccuracy score for DT : ", accuracy_score(y_test, y_pred_dt) * 100, "%")

    # Split the dataset into training and testing sets for word embeddings
    X_train, X_test, y_train, y_test = split_dataset(word_embeddings_feature_matrix, labels)
    # Classify the using Naive Bayes
    y_pred_nb = classifiers.classify_gaussian_naive_bayes(X_train, X_test, y_train)
    # Classify the using KNN
    y_pred_knn = classifiers.classify_knn(X_train, X_test, y_train, 100)
    # Classify the using SVM
    y_pred_svm = classifiers.classify_svm(X_train, X_test, y_train)
    # Classify using decision tree
    y_pred_dt = classifiers.classify_decision_tree(X_train, X_test, y_train)
    # Print the accuracy scores
    print("Word Embeddings")
    print("\tAccuracy score for GNB: ", accuracy_score(y_test, y_pred_nb) * 100, "%")
    print("\tAccuracy score for KNN: ", accuracy_score(y_test, y_pred_knn) * 100, "%")
    print("\tAccuracy score for SVM: ", accuracy_score(y_test, y_pred_svm) * 100, "%")
    print("\tAccuracy score for DT : ", accuracy_score(y_test, y_pred_dt) * 100, "%")

    # Close the database
    database.close_database(mydb)


if __name__ == '__main__':
    main()
