from src.preprocessing import load_and_preprocess
from src.train_knn import train_knn
from src.train_decision_tree import train_decision_tree
from src.train_naive_bayes import train_naive_bayes
from src.train_logistic_regression import train_logistic_regression
from src.train_random_forest import train_random_forest
from src.train_svm import train_svm
from src.train_kmeans import train_kmeans
from src.evaluate import evaluate_model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load original dataset for comparison and insights
df_original = pd.read_csv("heart.csv")
print("=== Basic Info ===")
df_original.info()
print("\n=== Statistical Summary ===")
print(df_original.describe())

df_vis = df_original.copy()

# ============================================
# 🚫 WRONG INTERPRETATION (initial assumption)
# ============================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df_vis, x="target", palette="coolwarm")
plt.title("Initial Interpretation of Target\n❌ WRONG Meaning")
plt.xlabel("Target Value")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Disease (0?)", "Disease (1?)"])
plt.tight_layout()
plt.show()

# ============================================
# ✅ CORRECT INTERPRETATION (fixed)
# ============================================

df_vis["disease_label"] = df_vis["target"].map({
    0: "Heart Disease",
    1: "No Heart Disease"
})
# Correct label mappings
df_vis["gender_label"] = df_vis["sex"].map({0: "Female", 1: "Male"})


plt.figure(figsize=(6, 4))
sns.countplot(data=df_vis, x="disease_label", palette="viridis")
plt.title("Correct Meaning of Target")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Load processed data for ML
df_processed, X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
sns.set(style="whitegrid")

# ------- After Capping Outliers -------
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_processed[col], color="#4C72B0")
    plt.title(f"Box Plot After Capping for {col}")
    plt.xlabel(col)
    plt.tight_layout()

# ------- Before Capping Outliers -------
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

# Scatter Plots
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_original, x="age", y="thalach", color="#55A868")
plt.title("Age vs Max Heart Rate")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_original, x="age", y="chol", color="#C44E52")
plt.title("Age vs Cholesterol")

# Count Plots
plt.figure(figsize=(7, 5))
sns.countplot(data=df_vis, x="gender_label", hue="disease_label", palette="Set2")
plt.title("Heart Disease by Gender (Correct Labels)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Condition")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df_original, x="target", color="#8172B2")
plt.title("Target Distribution")

plt.tight_layout()
plt.show()

# ============================
# 📌 MODEL TRAINING & EVALUATION
# ============================
from sklearn.model_selection import GridSearchCV

results = []

#KNN
knn_model = train_knn(X_train, y_train)
results = evaluate_model("KNN", knn_model, X_test, y_test, results)

# Decision Tree
dt_model = train_decision_tree(X_train, y_train)
results = evaluate_model("Decision Tree", dt_model, X_test, y_test, results)

# Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)
results = evaluate_model("Naive Bayes", nb_model, X_test, y_test, results)

# Logistic Regression
lr_model = train_logistic_regression(X_train, y_train)
results = evaluate_model("Logistic Regression", lr_model, X_test, y_test, results)

# Random Forest
rf_model = train_random_forest(X_train, y_train)
results = evaluate_model("Random Forest", rf_model, X_test, y_test, results)

# SVM
svm_model = train_svm(X_train, y_train)
results = evaluate_model("SVM", svm_model, X_test, y_test, results)

# K-Means
kmeans = train_kmeans(X_train, n_clusters=2)
results = evaluate_model("KMeans", kmeans, X_test, y_test, results)

results_df = pd.DataFrame(results)
print("\n=== Model Performance Comparison ===")
print(results_df)

plt.figure(figsize=(8, 5))
plt.bar(results_df["model"], results_df["accuracy"], color="skyblue")
plt.title("Accuracy Comparison of ML Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/model_scores.csv", index=False)
print("Model scores saved to results/model_scores.csv")

# ==========================
# Gender Risk Analysis (0 = heart disease)
# ==========================

# % of HEART DISEASE (target == 0) per gender
gender_stats = df_original.groupby("sex").apply(
    lambda g: (g["target"] == 0).mean() * 100
)

# Map index to labels (assuming 0=Female, 1=Male)
gender_stats.index = ["Female", "Male"]

print("\n--- Heart Disease Prevalence by Gender (%) ---")
print(gender_stats.round(2))

plt.figure(figsize=(6, 4))
sns.barplot(x=gender_stats.index, y=gender_stats.values, palette="coolwarm")
plt.ylabel("Heart Disease %")
plt.xlabel("Gender")
plt.title("Heart Disease Risk by Gender (0 = Disease)")
plt.tight_layout()
plt.show()


# ==========================
# Age Group Risk Analysis (0 = heart disease)
# ==========================

# Work on a copy so we don't permanently modify df_original if you don't want to
df_age = df_original.copy()

df_age["age_group"] = pd.cut(
    df_age["age"],
    bins=[20, 40, 50, 60, 80],
    labels=["20-40", "41-50", "51-60", "61-80"]
)

# % of HEART DISEASE (target == 0) per age group
age_risk = df_age.groupby("age_group").apply(
    lambda g: (g["target"] == 0).mean() * 100
)

print("\n--- Heart Disease Prevalence by Age Group (%) ---")
print(age_risk.round(2))

plt.figure(figsize=(7, 5))
sns.lineplot(x=age_risk.index, y=age_risk.values, marker="o", color="green")
plt.ylabel("Heart Disease %")
plt.xlabel("Age Group")
plt.title("Heart Disease Risk by Age Group (0 = Disease)")
plt.tight_layout()
plt.show()


# ====================================
# 📌 FEATURE IMPORTANCE (RANDOM FOREST)
# ====================================

import numpy as np

feature_importances = rf_model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(np.array(X_train.columns)[indices][:10],
        feature_importances[indices][:10], color="teal")
plt.title("Top 10 Important Features")
plt.ylabel("Feature Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n--- Top 10 Feature Importances ---")
for i in range(10):
    print(f"{X_train.columns[indices][i]}: {feature_importances[indices][i]:.4f}")


# ==================================
# MODEL DIAGNOSTIC VISUALIZATIONS
# ==================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN Elbow Curve
k_values = range(1, 21)
knn_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_scores, marker='o')
plt.title("KNN Accuracy vs K Value (Elbow Method)")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.tight_layout()
plt.show()

features = ["age", "trestbps", "chol", "thalach"]

for col in features:
    plt.figure(figsize=(8, 5))
    # 0 = heart disease
    sns.kdeplot(
        df_original[df_original["target"] == 0][col],
        label="Heart Disease (0)",
        fill=True,
        alpha=0.4
    )
    # 1 = no heart disease
    sns.kdeplot(
        df_original[df_original["target"] == 1][col],
        label="No Heart Disease (1)",
        fill=True,
        alpha=0.4
    )
    plt.title(f"Distribution of {col} by Condition (0 = Disease, 1 = No Disease)")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 5))
plt.bar(X_train.columns[indices], importances[indices])
plt.title("Top 10 Feature Importances (Decision Tree)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


depths = range(1, 21)
train_scores = []
test_scores = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, label="Train Accuracy")
plt.plot(depths, test_scores, label="Test Accuracy")
plt.title("Decision Tree: Depth vs Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 12))
plot_tree(dt, filled=True, feature_names=X_train.columns, max_depth=3)
plt.title("Decision Tree Structure (Depth=3)")
plt.show()

