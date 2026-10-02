import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve


def plot_learning_curve(
    model,
    X,
    y,
    cv=5
):

    train_sizes, train_scores, validation_scores = (
        learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            train_sizes=[
                0.1,
                0.25,
                0.5,
                0.75,
                1.0
            ]
        )
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        label="Training Score"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        label="Validation Score"
    )

    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")

    plt.legend()
    plt.grid(True)

    plt.show()
