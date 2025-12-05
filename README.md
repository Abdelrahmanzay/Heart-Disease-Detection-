# Heart Disease Detection

A comprehensive machine learning project for predicting heart disease using multiple classification algorithms and data preprocessing techniques.

## 📋 Overview

This project implements an end-to-end machine learning pipeline for heart disease detection. It includes:

- **Data Preprocessing**: Data cleaning, imputation, normalization, and outlier detection
- **Multiple ML Models**: Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, Naive Bayes, Random Forest, Support Vector Machine (SVM), and K-Means clustering
- **Model Evaluation**: Comprehensive metrics and performance analysis
- **Data Visualization**: Correlation analysis, distributions, and scatter plots

## 📁 Project Structure

```
Heart-Disease-Detection/
├── heart.csv                          # Dataset (305 samples, 13 features + target)
├── main.py                            # Main entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── src/                               # Source code
│   ├── preprocessing.py               # Data preprocessing and visualization
│   ├── train_decision_tree.py         # Decision Tree model
│   ├── train_knn.py                   # K-Nearest Neighbors model
│   ├── train_logistic_regression.py   # Logistic Regression model
│   ├── train_naive_bayes.py           # Naive Bayes model
│   ├── train_random_forest.py         # Random Forest model
│   ├── train_kmeans.py                # K-Means clustering model
│   ├── train_svm.py                   # Support Vector Machine model
│   └── evaluate.py                    # Model evaluation utilities
└── Requirements/                      # Additional documentation

```

## 📊 Dataset

The `heart.csv` dataset contains **305 samples** with the following features:

| Feature    | Description                             | Type            |
| ---------- | --------------------------------------- | --------------- |
| age        | Age of the patient                      | Numerical       |
| sex        | Gender (0=Female, 1=Male)               | Categorical     |
| cp         | Chest pain type (0-3)                   | Categorical     |
| trestbps   | Resting blood pressure                  | Numerical       |
| chol       | Serum cholesterol level                 | Numerical       |
| fbs        | Fasting blood sugar > 120 mg/dl         | Categorical     |
| restecg    | Resting ECG results                     | Categorical     |
| thalach    | Maximum heart rate achieved             | Numerical       |
| exang      | Exercise-induced angina                 | Categorical     |
| oldpeak    | ST depression induced by exercise       | Numerical       |
| slope      | Slope of ST segment                     | Categorical     |
| ca         | Number of major vessels                 | Categorical     |
| thal       | Thalassemia type                        | Categorical     |
| **target** | **Heart disease present (0=No, 1=Yes)** | **Categorical** |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip or conda

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Abdelrahmanzay/Heart-Disease-Detection-.git
cd Heart-Disease-Detection-
```

2. Create a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms

## 📖 Usage

### Step 1: Data Preprocessing

Run the preprocessing script to clean, normalize, and visualize the data:

```bash
python src/preprocessing.py
```

This will:

- Load and inspect the dataset
- Handle missing values (imputation)
- Normalize numerical features using MinMaxScaler
- Generate box plots for outlier detection
- Create correlation heatmaps and distribution plots

### Step 2: Train Models

Train individual models using the provided training scripts:

```bash
# Decision Tree
python src/train_decision_tree.py

# K-Nearest Neighbors
python src/train_knn.py

# Logistic Regression
python src/train_logistic_regression.py

# Naive Bayes
python src/train_naive_bayes.py

# Random Forest
python src/train_random_forest.py

# Support Vector Machine
python src/train_svm.py

# K-Means Clustering
python src/train_kmeans.py
```

### Step 3: Model Evaluation

Evaluate all models:

```bash
python src/evaluate.py
```

### Run All at Once

Execute the main pipeline:

```bash
python main.py
```

## 🔬 Machine Learning Models

### Supervised Learning Models

- **Logistic Regression**: Linear classifier for binary classification
- **Decision Tree**: Tree-based model for interpretability
- **Random Forest**: Ensemble method using multiple decision trees
- **K-Nearest Neighbors (KNN)**: Instance-based learning algorithm
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Support Vector Machine (SVM)**: Kernel-based classifier

### Unsupervised Learning

- **K-Means Clustering**: Partitioning-based clustering algorithm

## 📈 Key Features

✅ **Data Preprocessing**

- Missing value imputation (mean for numerical, mode for categorical)
- Feature normalization (MinMaxScaler)
- Outlier detection using box plots

✅ **Visualization**

- Correlation heatmaps
- Distribution plots
- Scatter plots (Age vs Heart Rate, Age vs Cholesterol)
- Count plots for categorical variables

✅ **Model Training**

- Train-test split validation
- Cross-validation support
- Hyperparameter tuning capabilities

✅ **Evaluation Metrics**

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Classification Reports

## 🛠️ Troubleshooting

### Import Errors

If you see `ImportError: No module named 'seaborn'` or similar:

```bash
pip install --upgrade -r requirements.txt
```

### Data File Not Found

Ensure `heart.csv` is in the project root directory or update the file path in the scripts.


## 📊 Example Output

After running preprocessing, you'll see:

- Dataset shape and basic statistics
- Preprocessed data preview
- Box plots for numerical features
- Correlation heatmap
- Scatter plots of feature relationships
- Count plots for categorical distributions




**Last Updated**: December 2025
