import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec


def apply_tfidf_vectorizer(features):
    """
    Apply TF-IDF vectorizer
    :param features: Features
    :return: TF-IDF feature matrix
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_description = vectorizer.fit_transform(features['description'].fillna(''))
    feature_name = vectorizer.fit_transform(features['name'].fillna(''))
    feature_provider = vectorizer.fit_transform(features['provider'].fillna(''))
    feature_tags = vectorizer.fit_transform(features['tags'].fillna(''))
    feature_summary = vectorizer.fit_transform(features['summary'].fillna(''))
    feature_matrix = hstack([feature_description, feature_name, feature_tags, feature_summary, feature_provider])
    return feature_matrix


def apply_word_embeddings(features):
    """
    Apply word embeddings
    :param features: Features
    :return: Word embeddings feature matrix
    """
    # concatenate all tokenized text into a list
    tokenized_text = features['description'].fillna('').tolist() + features['name'].fillna('').tolist() + \
                     features['provider'].fillna('').tolist() + \
                     features['tags'].fillna('').tolist() + features['summary'].fillna('').tolist()

    # train word2vec model
    model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

    # Create an empty feature matrix
    feature_matrix = np.zeros((len(features), model.vector_size * 5))

    # Iterate over each row
    for i, row in features.iterrows():
        # print(i)
        # if i == 479:
        #     print('here')
        if len(row['description']) == 0:
            # print('description is empty')
            row['description'] = row['name']
        if len(row['provider']) == 0:
            # print('provider is empty')
            row['provider'] = row['name']
        if len(row['tags']) == 0:
            # print('tags is empty')
            row['tags'] = row['name']
        feature_vector = np.concatenate(
            [
                np.mean([model.wv[word] for word in row['description']], axis=0),
                np.mean([model.wv[word] for word in row['name']], axis=0),
                np.mean([model.wv[word] for word in row['provider']] if len(row['provider']) != 0 else np.zeros(
                    model.vector_size), axis=0),
                np.mean([model.wv[word] for word in row['tags']], axis=0),
                np.mean([model.wv[word] for word in row['summary']], axis=0)
            ]
        )

        feature_matrix[i, :] = feature_vector

    # normalize the feature matrix
    feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1)[:, np.newaxis]

    return feature_matrix


def transform_to_non_negative(word_embeddings_feature_matrix):
    """
    Transform the word embeddings feature matrix to non-negative values
    :param word_embeddings_feature_matrix: Word embeddings feature matrix
    :return: Non-negative word embeddings feature matrix
    """
    return word_embeddings_feature_matrix - np.min(word_embeddings_feature_matrix, axis=0)


def apply_tfidf_transformer(word_embeddings_feature_matrix):
    """
    Apply TF-IDF transformer
    :param word_embeddings_feature_matrix: Word embeddings feature matrix
    :return: TF-IDF transformed feature matrix
    """
    transformer = TfidfVectorizer(stop_words='english')
    feature_matrix = transformer.fit_transform(word_embeddings_feature_matrix)
    return feature_matrix
