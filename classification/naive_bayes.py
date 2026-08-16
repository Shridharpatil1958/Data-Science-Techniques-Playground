"""
Naive Bayes Classifier (Gaussian)
====================================
Usage:
    python classification/naive_bayes.py
"""

from sklearn.naive_bayes import GaussianNB
from _common import load_and_prepare, evaluate_classifier


def main():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    evaluate_classifier(model, X_test_scaled, y_test, model_name="Gaussian Naive Bayes")
    return model


if __name__ == "__main__":
    main()
