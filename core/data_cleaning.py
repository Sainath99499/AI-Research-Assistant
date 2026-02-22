import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df, target_column):
    df = df.drop_duplicates()
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y