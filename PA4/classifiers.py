from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import evaluation


def classify_using_multinomial_naive_bayes(X_train, X_test, y_train):
    """
    Classify the dataset using Naive Bayes
    :param X_train: training set
    :param X_test: testing set
    :param y_train: training labels
    :return: predicted labels
    """
    # Create a naive bayes classifier
    naive_bayes_classifier = MultinomialNB(alpha=0.5)
    # eval_score = evaluation.evaluate_classifier(naive_bayes_classifier, X_train, y_train)
    # print(naive_bayes_classifier, "CV score: ", eval_score)
    naive_bayes_classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = naive_bayes_classifier.predict(X_test)

    return y_pred


def classify_using_gaussian_naive_bayes(X_train, X_test, y_train):
    """
    Classify the dataset using Naive Bayes
    :param X_train: training set
    :param X_test: testing set
    :param y_train: training labels
    :return: predicted labels
    """
    # Create a naive bayes classifier
    naive_bayes_classifier = GaussianNB(var_smoothing=1e-17)
    # eval_score = evaluation.evaluate_classifier(naive_bayes_classifier, X_train, y_train)
    # print(naive_bayes_classifier, "CV score: ", eval_score)
    naive_bayes_classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = naive_bayes_classifier.predict(X_test)

    return y_pred


def classify_using_knn(X_train, X_test, y_train, N_NEIGHBORS=200):
    """
    Classify the dataset using KNN
    :param N_NEIGHBORS: number of neighbors
    :param X_train: training set
    :param X_test: testing set
    :param y_train: training labels
    :return: predicted labels
    """
    # Create a nearest neighbors classifier
    nearest_neighbors_classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    # eval_score = evaluation.evaluate_classifier(nearest_neighbors_classifier, X_train, y_train)
    # print(nearest_neighbors_classifier, "CV score: ", eval_score)
    nearest_neighbors_classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = nearest_neighbors_classifier.predict(X_test)

    return y_pred


def classify_using_svm(X_train, X_test, y_train):
    """
    Classify the dataset using SVM
    :param X_train: training set
    :param X_test: testing set
    :param y_train: training labels
    :return: predicted labels
    """
    # Create a support vector machine classifier
    svm_classifier = SVC(kernel='linear', C=0.5)
    # eval_score = evaluation.evaluate_classifier(svm_classifier, X_train, y_train)
    # print(svm_classifier, "CV score: ", eval_score)
    svm_classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = svm_classifier.predict(X_test)

    return y_pred


def classify_using_decision_tree(X_train, X_test, y_train):
    """
   Classify the dataset using decision tree
   :param X_train: training set
   :param X_test: testing set
   :param y_train: training labels
   :return: predicted labels
    """
    # Create a decision tree classifier
    decision_tree_classifier = DecisionTreeClassifier(max_depth=20)
    # eval_score = evaluation.evaluate_classifier(decision_tree_classifier, X_train, y_train)
    # print(decision_tree_classifier, "CV score: ", eval_score)
    decision_tree_classifier.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = decision_tree_classifier.predict(X_test)

    return y_pred
