from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=15,
    random_state=42
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model



def train_xgboost(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
):

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    return model



def train_lightgbm(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
):

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train
    )

    return model