import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from an ensemble model.
    """

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance


def create_shap_explainer(model):
    """
    Create a SHAP explainer.
    """

    return shap.Explainer(model)

# def create_shap_explainer(
#     model,
#     X_train,
#     sample_size=2000,
#     random_state=42
# ):
#     """
#     Create a SHAP TreeExplainer using
#     a sample of the training data.
#     """

#     if len(X_train) > sample_size:
#         X_sample = X_train.sample(
#             sample_size,
#             random_state=random_state
#         )
#     else:
#         X_sample = X_train

#     # explainer = shap.TreeExplainer(model)

#     # shap_values = explainer.shap_values(X_sample)
#     explainer = shap.Explainer(model)

#     shap_values = explainer(X_sample)
#     shap_values.shape
#     return explainer, shap_values, X_sample

def get_prediction_examples(
    model,
    X_test,
    y_test
):
    """
    Return one TP, FP and FN index.
    Returns None if a category does not exist.
    """

    y_pred = model.predict(X_test)

    tp = np.where((y_test == 1) & (y_pred == 1))[0]
    fp = np.where((y_test == 0) & (y_pred == 1))[0]
    fn = np.where((y_test == 1) & (y_pred == 0))[0]

    return (
        tp[0] if len(tp) > 0 else None,
        fp[0] if len(fp) > 0 else None,
        fn[0] if len(fn) > 0 else None,
    )
def plot_feature_importance_comparison(
    fraud_importance,
    credit_importance,
    top_n=10
):
    """
    Plot the top N feature importances for the
    Fraud Transaction and Credit Card datasets
    in a single figure.
    """

    fraud_top = fraud_importance.head(top_n)
    credit_top = credit_importance.head(top_n)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    # Fraud Dataset
    axes[0].barh(
        fraud_top["Feature"][::-1],
        fraud_top["Importance"][::-1]
    )
    axes[0].set_title("Fraud Transaction Dataset\nLightGBM")
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_ylabel("Feature")

    # Credit Card Dataset
    axes[1].barh(
        credit_top["Feature"][::-1],
        credit_top["Importance"][::-1]
    )
    axes[1].set_title("Credit Card Dataset\nRandom Forest")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_ylabel("Feature")

    plt.suptitle(
        "Top 10 Feature Importance Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

def plot_shap_summary(
    shap_values,
    X,
    title
):
    """
    Plot SHAP summary plot.
    """

    plt.figure(figsize=(10, 6))

    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    if values.ndim == 3:
        values = values[:, :, 1]

    shap.summary_plot(
        values,
        X,
        max_display=10,
        show=False
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

# def plot_shap_summary(
#     shap_values,
#     title,
#     max_display=10
# ):
#     """
#     Plot SHAP summary plot using the new SHAP API.
#     """

#     # plt.figure(figsize=(10, 6))

#     shap.summary_plot(
#         shap_values.values,
#         features=shap_values.data,
#         feature_names=shap_values.feature_names,
#         max_display=max_display,
#         show=False
#     )

#     plt.title(title, fontsize=14)
#     plt.tight_layout()
#     plt.show()
    
def plot_shap_summary_comparison(
    fraud_explainer,
    X_train_fraud,
    credit_explainer,
    X_train_cc
):
    """
    Plot SHAP summary plots for the Fraud Transaction
    and Credit Card datasets side by side.
    """

    fraud_shap = fraud_explainer(X_train_fraud)
    credit_shap = credit_explainer(X_train_cc)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7)
    )

    # Fraud Dataset
    plt.sca(axes[0])
    shap.summary_plot(
        fraud_shap,
        X_train_fraud,
        max_display=10,
        show=False
    )
    axes[0].set_title(
        "Fraud Transaction Dataset\nLightGBM"
    )

    # Credit Card Dataset
    plt.sca(axes[1])
    shap.summary_plot(
        credit_shap,
        X_train_cc,
        max_display=10,
        show=False
    )
    axes[1].set_title(
        "Credit Card Dataset\nRandom Forest"
    )

    plt.suptitle(
        "SHAP Summary Plot Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()

import shap

def plot_force_plot(
    shap_values,
    index,
    title
):
    """
    Display SHAP force plot in VS Code Notebook.
    """

    print(title)

    shap.initjs()

    force = shap.plots.force(
        shap_values[index]
    )

    # display(force)

    return force