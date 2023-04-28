import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def apply_tfidf_vectorizer(dataset):
    """
    Apply TF-IDF vectorizer
    :param dataset: dataset read from mongodb
    :return: TF-IDF feature matrix
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix_tfidf = vectorizer.fit_transform(dataset['text'].fillna(''))
    return feature_matrix_tfidf


def apply_lda(tfidf_features):
    """
    Apply LDA
    :param tfidf_features: TF-IDF feature matrix
    :param dataset: dataset read from mongodb
    :return: LDA feature matrix
    """
    lda = LatentDirichletAllocation(n_components=50, random_state=42)
    feature_matrix_lda = lda.fit_transform(tfidf_features)
    return feature_matrix_lda


def apply_word_embeddings(dataset):
    """
    Apply word embeddings
    :param dataset: dataset read from mongodb
    :return: Word embeddings feature matrix
    """
    corpus = [text.split() for text in dataset['text'].fillna('')]
    tokenized_text = [text for text in corpus if text != []]

    # train word2vec model
    model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

    # Create word embeddings
    feature_matrix_word2vec = np.zeros((len(tokenized_text), 100))
    for i in range(len(tokenized_text)):
        feature_matrix_word2vec[i] = np.mean(model.wv[tokenized_text[i]], axis=0)

    return feature_matrix_word2vec


def extract_features(dataset):
    """
    Extract features from the dataset
    :param dataset: dataset read from mongodb
    :return: features using 3 different models for feature extraction, viz. tfidf, word2vec and LDA
    """
    # Preprocessing
    dataset['text'] = dataset['description'] + ' ' + dataset['name'] + ' ' \
                      + dataset['summary'] + ' ' + dataset['tags'] + ' ' + dataset['provider']
    dataset = dataset.drop(['description', 'name', 'summary', 'tags', 'provider'], axis=1)

    # tfidf
    tfidf_features = apply_tfidf_vectorizer(dataset)

    # word2vec
    word2vec_features = apply_word_embeddings(dataset)

    # LDA
    lda_features = apply_lda(tfidf_features)

    return tfidf_features, word2vec_features, lda_features
