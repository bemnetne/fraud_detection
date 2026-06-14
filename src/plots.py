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
def plot_confusion_matrices_tuned(
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

    axes[0].set_title("Fraud Dataset(LightGBM)")
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

    axes[1].set_title("Credit Card Dataset(Random Forest)")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("Actual Class")

    plt.suptitle(
        "Confusion Matrices for tuned models",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()
def plot_confusion_matricess(
    y_test_fraud,
    rf_pred_fraud,
    xgb_pred_fraud,
    lgbm_pred_fraud,
    y_test_credit,
    rf_pred_credit,
    xgb_pred_credit,
    lgbm_pred_credit
):
    """
    Plot confusion matrices for Random Forest,
    XGBoost and LightGBM on both datasets.
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    fraud_models = [
        ("Random Forest", rf_pred_fraud),
        ("XGBoost", xgb_pred_fraud),
        ("LightGBM", lgbm_pred_fraud)
    ]

    credit_models = [
        ("Random Forest", rf_pred_credit),
        ("XGBoost", xgb_pred_credit),
        ("LightGBM", lgbm_pred_credit)
    ]

    # Fraud Dataset
    for i, (name, pred) in enumerate(fraud_models):

        cm = confusion_matrix(
            y_test_fraud,
            pred
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axes[0, i]
        )

        axes[0, i].set_title(f"Fraud Dataset\n{name}")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("Actual")

    # Credit Card Dataset
    for i, (name, pred) in enumerate(credit_models):

        cm = confusion_matrix(
            y_test_credit,
            pred
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axes[1, i]
        )

        axes[1, i].set_title(f"Credit Card Dataset\n{name}")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("Actual")

    plt.suptitle(
        "Confusion Matrices for Ensemble Models",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()