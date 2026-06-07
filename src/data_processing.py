import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler

def handle_missing_values(
    df,
    numerical_cols=None,
    categorical_cols=None
):
    """
    Impute missing values:
    - Median for numerical columns
    - Mode for categorical columns
    """

    df = df.copy()

    numerical_cols = numerical_cols or []
    categorical_cols = categorical_cols or []

    # Missing values before
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before imputation: {missing_before}")

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(
                df[col].median()
            )

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicates}")

    df = df.drop_duplicates()

    return df


def correct_data_types(
    df,
    datetime_cols=None,
    ip_col=None
):
    """
    Correct column data types.
    """

    df = df.copy()

    datetime_cols = datetime_cols or []

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if ip_col and ip_col in df.columns:
        df[ip_col] = (pd.to_numeric(
            df[ip_col],
            errors="coerce"
    )
    .fillna(0)
    .astype("int64")
        )
    return df


def data_quality_report(df):
    """
    Generate a simple data quality report.
    """

    report = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Missing (%)": (
            df.isnull().mean() * 100
        ).round(2),
        "Unique Values": df.nunique()
    })

    return report


def clean_data(
    df,
    numerical_cols=None,
    categorical_cols=None,
    datetime_cols=None,
    ip_col=None
):
    """
    Complete cleaning pipeline.
    """

    df = handle_missing_values(
        df,
        numerical_cols,
        categorical_cols
    )

    df = remove_duplicates(df)

    df = correct_data_types(
        df,
        datetime_cols,
        ip_col
    )

    return df

def class_distribution(df, target_col):
    """
    Display class counts and percentages.
    """

    counts = df[target_col].value_counts()
    percentages = round(
        df[target_col].value_counts(normalize=True) * 100,
        2
    )

    result = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages
    })

    print(result)

    return result


def plot_numerical_distributions(df, columns):
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(12, 5)
    )

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.histplot(
            df[col],
            kde=True,
            ax=ax
        )
        ax.set_title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df, columns):
    
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(15, 5)
    )

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        
        sns.countplot(
            data=df,
            x=col,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {col}"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

    plt.tight_layout()
    plt.show()


def fraud_boxplots(
    df,
    numerical_cols,
    target_col
):

    n_cols = 2
    n_rows = math.ceil(
        len(numerical_cols) / n_cols
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 5 * n_rows)
    )

    axes = axes.flatten()

    for i, col in enumerate(
        numerical_cols
    ):

        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            ax=axes[i]
        )

        axes[i].set_title(
            f"{col} vs {target_col}"
        )

    # Remove unused subplots
    for j in range(
        len(numerical_cols),
        len(axes)
    ):
        fig.delaxes(
            axes[j]
        )

    plt.tight_layout()
    plt.show()


def fraud_rate_by_category(
    df,
    categorical_cols,
    target_col
):

    if isinstance(categorical_cols, str):
        categorical_cols = [categorical_cols]

    fraud_rates_list = []

    for col in categorical_cols:

        rates = (
            df.groupby(col)[target_col]
            .mean()
            .reset_index()
        )

        rates.columns = [
            "category",
            "fraud_rate"
        ]

        rates["feature"] = col

        fraud_rates_list.append(
            rates
        )

    fraud_rates = pd.concat(
        fraud_rates_list,
        ignore_index=True
    )

    plt.figure(figsize=(14, 6))

    sns.barplot(
        data=fraud_rates,
        x="category",
        y="fraud_rate",
        hue="feature"
    )

    plt.title(
        "Fraud Rate by Category"
    )

    plt.ylabel(
        "Fraud Rate"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return fraud_rates

def correlation_heatmap(df):

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.show()

def plot_class_distribution(
    df,
    target_col
):

    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=df,
        x=target_col
    )

    plt.title(
        f"Class Distribution: {target_col}"
    )

    plt.show()

    percentages = (
        df[target_col]
        .value_counts(normalize=True)
        * 100
    )

    print("\nClass Percentages:")
    print(percentages)

def prepare_ip_country_data(ip_country_df):

    ip_country_df = ip_country_df.copy()

    ip_country_df["lower_bound_ip_address"]=(pd.to_numeric(
            ip_country_df["lower_bound_ip_address"],
            errors="coerce"
    )
    .fillna(0)
    .astype("int64")
        )
    ip_country_df["upper_bound_ip_address"]=(pd.to_numeric(
            ip_country_df["upper_bound_ip_address"],
            errors="coerce"
    )
    .fillna(0)
    .astype("int64")
        )

    return ip_country_df

def merge_country_info(
    fraud_df,
    ip_country_df
):
    """
    Assign country to each transaction
    based on IP range.
    """

    fraud_df = fraud_df.sort_values(
        "ip_address"
    )

    ip_country_df = ip_country_df.sort_values(
        "lower_bound_ip_address"
    )

    merged_df = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    merged_df = merged_df[
        merged_df["ip_address"]
        <= merged_df["upper_bound_ip_address"]
    ]

    return merged_df

def fraud_rate_by_country(df):

    country_fraud = (
        df.groupby("country")["class"]
        .agg(
            total_transactions="count",
            fraud_transactions="sum",
            fraud_rate="mean"
        )
        .sort_values(
            "fraud_rate",
            ascending=False
        )
    )

    country_fraud["fraud_rate"] *= 100

    return country_fraud

import pandas as pd


def create_fraud_features(df):
    """
    Create engineered features for Fraud_Data.csv
    """

    df = df.copy()


    # Time-based features
    df["hour_of_day"] = df["purchase_time"].dt.hour

    df["day_of_week"] = (
        df["purchase_time"]
        .dt.day_name()
    )

    # Time since signup (hours)
    df["time_since_signup_hours"] = (
        (
            df["purchase_time"]
            - df["signup_time"]
        )
        .dt.total_seconds()
        / 3600
    )

    # Time since signup (days)
    df["time_since_signup_days"] = (
        (
            df["purchase_time"]
            - df["signup_time"]
        )
        .dt.total_seconds()
        / (3600 * 24)
    )

    return df
def add_transaction_frequency(df):

    df = df.copy()

    df["user_transaction_count"] = (
        df.groupby("user_id")["user_id"]
        .transform("count")
    )

    return df

def add_transaction_velocity(df):
    """
    Number of previous transactions made by a user.
    """

    df = df.copy()

    df = df.sort_values(
        ["user_id", "purchase_time"]
    )

    df["transactions_before"] = (
        df.groupby("user_id")
        .cumcount()
    )

    return df

def feature_engineering(df):

    df = create_fraud_features(df)

    df = add_transaction_frequency(df)

    df = add_transaction_velocity(df)

    return df


def scale_numerical_features(
    df,
    numerical_cols
):
    """
    Standardize numerical features.
    """

    df = df.copy()

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    return df, scaler

def encode_categorical_features(
    df,
    categorical_cols
):
    """
    One-hot encode categorical features.
    """

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    return df