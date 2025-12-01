import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(
    csv_path="heart.csv",
    output_csv="data/heart_preprocessed.csv",
    test_size=0.2,
    random_state=42
):
    df = pd.read_csv("heart.csv")
    print("✅ Dataset loaded successfully!\n")

    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    target_col = "target"

    n_dup = df.duplicated().sum()
    print("Exact duplicate rows:", n_dup)

    dupes = df[df.duplicated(keep=False)]
    print(dupes)

    df = df.drop_duplicates()
    print("Duplicates removed and heart.csv overwritten.")

    # Handle missing values
    print("=== Missing Values Count ===")
    print(df.isnull().sum())

    if df.isnull().sum().sum() > 0:
        print("\n⚠ Missing values found — applying SimpleImputer...")
        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print("✔ Missing values imputed.\n")
    else:
        print("\n✔ No missing values — skipping imputation.\n")

    print("\n=== Preview of Preprocessed Data ===")
    print(df.head())

    #outlier removal
    print("\n=== Detecting and Capping Outliers ===")

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"\nColumn: {col}")
        print("Outliers found:", len(outliers))
        df[col] = np.where(df[col] < lower, lower,np.where(df[col] > upper, upper, df[col]))
                                                
    print("\n✔ Outliers capped successfully!\n")

    df_processed_for_vis = df.copy()

    #One-hot encoding 
    X = pd.get_dummies(df.drop(columns=[target_col]), columns=cat_cols, drop_first=True)
    y = df[target_col]
    # Normalize numerical features
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("\n=== Final Preprocessed Data Preview ===")
    print(df.head())

    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save final preprocessed data (for reference & plotting)
    df_processed_for_vis.to_csv(output_csv, index=False)

    return df_processed_for_vis, X_train, X_test, y_train, y_test, scaler