"""
Support Vector Machine Classifier
====================================
Usage:
    python classification/svm_classifier.py
"""

from sklearn.svm import SVC
from _common import load_and_prepare, evaluate_classifier


def main(kernel: str = "rbf", C: float = 1.0):
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare()

    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    evaluate_classifier(model, X_test_scaled, y_test, model_name=f"SVM ({kernel} kernel)")
    return model


if __name__ == "__main__":
    main()
