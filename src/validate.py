import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

def cross_validate_model(
    model,
    X,
    y,
    cv=5
):
    """
    Perform Stratified K-Fold Cross Validation.
    """

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "auc_pr": "average_precision"
    }

    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )

    results = {
        "F1 Mean": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std(),

        "Precision Mean": scores["test_precision"].mean(),
        "Precision Std": scores["test_precision"].std(),

        "Recall Mean": scores["test_recall"].mean(),
        "Recall Std": scores["test_recall"].std(),

        "AUC-PR Mean": scores["test_auc_pr"].mean(),
        "AUC-PR Std": scores["test_auc_pr"].std()
    }

    return results