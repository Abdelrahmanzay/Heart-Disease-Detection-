import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
 
df = pd.read_csv("heart.csv")

print("✅ Dataset loaded successfully!\n")

print("=== Basic Info ===")

df.info()

print("\n=== Statistical Summary ===")

print(df.describe())
 

# Separate numerical and categorical features
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]


#remove duplicates
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


# outliers visualization after capping 
sns.set(style="whitegrid")
 
for col in num_cols:
    plt.figure(figsize=(6, 4))

    sns.boxplot(x=df[col], color="#4C72B0")

    plt.title(f"Box Plot After Capping for {col}")

    plt.xlabel(col)

    plt.tight_layout()

# outliers visualization before capping 
df_original = pd.read_csv("heart.csv")
sns.set(style="whitegrid")
 
for col in num_cols:
    plt.figure(figsize=(6, 4))

    sns.boxplot(x=df_original[col], color="#4C72B0")

    plt.title(f"Box Plot for {col}")

    plt.xlabel(col)

    plt.tight_layout()


# Correlation Heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(df_original.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

plt.tight_layout()
 
# Scatter plots

plt.figure(figsize=(6, 4))

sns.scatterplot(data=df_original, x="age", y="thalach", color="#55A868")

plt.title("Age vs Max Heart Rate")
 
plt.figure(figsize=(6, 4))

sns.scatterplot(data=df_original, x="age", y="chol", color="#C44E52")

plt.title("Age vs Cholesterol")
 
# Count plots
 
plt.figure(figsize=(6, 4))

sns.countplot(data=df_original, x="sex", hue="target", palette="Set2")

plt.title("Sex Distribution with Target")
 
plt.figure(figsize=(6, 4))

sns.countplot(data=df_original, x="target", color="#8172B2")

plt.title("Target Variable Distribution")


plt.figure(figsize=(7, 5))
sns.countplot(data=df_original, x="sex", hue="target", palette="Set2")
plt.xticks([0, 1], ["Female", "Male"])
plt.title("Heart Disease by Gender (Grouped Bar Chart)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()

plt.figure(figsize=(8, 5))
sns.histplot(df_original[df_original["target"] == 0]["age"], kde=True, stat="density",
             alpha=0.5, label="No Heart Disease", bins=20)
sns.histplot(df_original[df_original["target"] == 1]["age"], kde=True, stat="density",
             alpha=0.5, label="Heart Disease", bins=20)
plt.title("Age Distribution: Heart Disease vs No Disease")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.show()

# Normalize numerical features
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


 