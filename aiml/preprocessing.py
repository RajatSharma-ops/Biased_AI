import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_dataset(dataset_path, target_col, sensitive_col, test_size=0.3):
    df = pd.read_csv(dataset_path)
    df = df.replace('?', pd.NA).dropna()

    # Check columns exist
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset.")

    # Encode target column to 0/1
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col])

    # Encode sensitive column to integers
    le_sensitive = LabelEncoder()
    A = le_sensitive.fit_transform(df[sensitive_col])

    # Separate features
    X = df.drop(columns=[target_col, sensitive_col])

    # Encode categorical features
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return train_test_split(X, y, A, test_size=test_size, random_state=42)