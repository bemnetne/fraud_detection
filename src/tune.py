from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using GridSearchCV.
    """
    param_grid = {
    "n_estimators": [200],
    "max_depth": [20]
    }
    # param_grid = {
    #     "n_estimators": [100, 200],
    #     "max_depth": [10, 20],
    #     "min_samples_split": [2, 5]
    # }

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best F1 Score:", grid.best_score_)

    return grid.best_estimator_

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost using GridSearchCV.
    """

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }

    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best F1 Score:", grid.best_score_)

    return grid.best_estimator_

def tune_lightgbm(X_train, y_train):
    """
    Tune LightGBM using GridSearchCV.
    """

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "learning_rate": [0.05, 0.1]
    }

    lgbm = LGBMClassifier(
        random_state=42,
        verbose=-1
    )

    grid = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best F1 Score:", grid.best_score_)

    return grid.best_estimator_