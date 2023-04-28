from sklearn.model_selection import cross_val_score


def evaluate_classifier(classifier, X, y, cvf=5):
    """
    Evaluate a classifier
    :param classifier: classifier
    :param X: features
    :param y: labels
    :param cvf: cross validation folds
    :return: mean accuracy
    """
    scores = cross_val_score(classifier, X, y, cv=cvf)
    return scores.mean()
