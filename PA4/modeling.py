import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import LatentDirichletAllocation
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


def apply_lda_topic_modeling(features):
    """
    Apply LDA topic modeling
    :param features: Features
    :return: LDA topic modeling feature matrix
    """
    lda = LatentDirichletAllocation(n_components=100, random_state=0)
    feature_matrix = lda.fit_transform(apply_tfidf_vectorizer(features))
    return feature_matrix


def get_feature_matrices(features):
    """
    Get feature matrices
    :param features: Features
    :return: Feature matrices
    """
    tfidf_feature_matrix = apply_tfidf_vectorizer(features)
    word_embeddings_feature_matrix = apply_word_embeddings(features)
    lda_feature_matrix = apply_lda_topic_modeling(features)
    return tfidf_feature_matrix, word_embeddings_feature_matrix, lda_feature_matrix
