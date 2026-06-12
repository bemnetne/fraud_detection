import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix - Logistic Regression")

    plt.show()

def plot_confusion_matrices(
    y_test_fraud,
    y_pred_fraud,
    y_test_credit,
    y_pred_credit
):
    """
    Plot confusion matrices for the Fraud and
    Credit Card datasets side by side.
    """

    cm_fraud = confusion_matrix(
        y_test_fraud,
        y_pred_fraud
    )

    cm_credit = confusion_matrix(
        y_test_credit,
        y_pred_credit
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    # Fraud Dataset
    sns.heatmap(
        cm_fraud,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=axes[0]
    )

    axes[0].set_title("Fraud Dataset")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("Actual Class")

    # Credit Card Dataset
    sns.heatmap(
        cm_credit,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=axes[1]
    )

    axes[1].set_title("Credit Card Dataset")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("Actual Class")

    plt.suptitle(
        "Confusion Matrices for Logistic Regression",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()