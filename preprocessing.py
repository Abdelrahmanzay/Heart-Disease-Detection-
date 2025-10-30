"""

Phase 1: Data Preprocessing, Outlier Detection, and Visualization for heart.csv
 
Required libraries:

- pandas, numpy

- matplotlib.pyplot, seaborn

- sklearn.impute.SimpleImputer

- sklearn.preprocessing.MinMaxScaler
 
Steps:

1. Load and inspect dataset

2. Preprocess data (handle missing values + scale features)

3. Detect outliers (box plots)

4. Visualize correlations and distributions

"""
 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
 
 
# ---------------------------------------

# Step 1: Load and Inspect Data

# ---------------------------------------

df = pd.read_csv("heart.csv")

print("✅ Dataset loaded successfully!\n")

print("=== Basic Info ===")

df.info()

print("\n=== Statistical Summary ===")

print(df.describe())
 
# ---------------------------------------

# Step 2: Data Preprocessing

# ---------------------------------------
 
# Separate numerical and categorical features

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
 
# Handle missing values

num_imputer = SimpleImputer(strategy="mean")

cat_imputer = SimpleImputer(strategy="most_frequent")
 
df[num_cols] = num_imputer.fit_transform(df[num_cols])

df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
 
# Normalize numerical features

scaler = MinMaxScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])
 
print("\n=== Preview of Preprocessed Data ===")

print(df.head())
 
# ---------------------------------------

# Step 3: Outlier Detection (Box Plots)

# ---------------------------------------

sns.set(style="whitegrid")
 
for col in ["trestbps", "chol", "thalach"]:

    plt.figure(figsize=(6, 4))

    sns.boxplot(x=df[col], color="#4C72B0")

    plt.title(f"Box Plot for {col}")

    plt.xlabel(col)

    plt.tight_layout()
 
# ---------------------------------------

# Step 4: Visualization & Correlation

# ---------------------------------------
 
# Correlation Heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

plt.tight_layout()
 
# Scatter plots

plt.figure(figsize=(6, 4))

sns.scatterplot(data=df, x="age", y="thalach", color="#55A868")

plt.title("Age vs Max Heart Rate")
 
plt.figure(figsize=(6, 4))

sns.scatterplot(data=df, x="age", y="chol", color="#C44E52")

plt.title("Age vs Cholesterol")
 
# Count plots

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="cp", color="#4C72B0")

plt.title("Distribution of Chest Pain Types")
 
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="sex", hue="target", palette="Set2")

plt.title("Sex Distribution with Target")
 
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="target", color="#8172B2")

plt.title("Target Variable Distribution")
 
plt.show()

 