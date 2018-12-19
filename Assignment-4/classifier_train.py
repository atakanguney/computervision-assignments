from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def construct_train_test(positive_features, negative_features):
    positive_labels = [1] * len(positive_features)
    negative_labels = [0] * len(negative_features)

    X = positive_features + negative_features
    y = positive_labels + negative_labels

    return train_test_split(X, y, test_size=0.2)


def test_score(y_pred, y_test):
    precision, recall, fscore, support = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F-Score: {}".format(fscore))
    print("Support: {}".format(support))
    print("Accuracy: {}".format(accuracy))


def train_classifier(positive_features, negative_features):

    X_train, X_test, y_train, y_test = construct_train_test(positive_features, negative_features)

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    test_score(y_pred, y_test)

    return classifier
