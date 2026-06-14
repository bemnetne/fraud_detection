from sklearn.model_selection import train_test_split
from src.data_processing import scale_numerical_features,encode_categorical_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score
)

def drop_unused_columns(df, columns_to_drop):
    """
    Drop identifier and raw timestamp columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns_to_drop (list): List of columns to remove.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """

    df = df.copy()

    # Only drop columns that exist
    existing_columns = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df = df.drop(columns=existing_columns)

    print("Dropped columns:", existing_columns)

    return df
def prepare_train_test_data(
    df,
    target_col,
    test_size=0.2,
    random_state=42
):
    """
    Split data into train and test sets
    while preserving class distribution.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def transform_data(
    df,
    numerical_cols,
    categorical_cols
):

    df = encode_categorical_features(
        df,
        categorical_cols
    )
    df, scaler = scale_numerical_features(
        df,
        numerical_cols
    )

    return df, scaler




def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression baseline model.
    """

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model using
    AUC-PR, F1-Score and Confusion Matrix.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(
        y_test,
        y_prob
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )
    ps = precision_score(
            y_test,
            y_pred
        )

    print(f"AUC-PR : {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix")
    print(cm)
    print(ps)
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return {
        "AUC-PR": auc_pr,
        "F1-Score": f1,
        "Confusion Matrix": cm,
        "Precision":ps
    }