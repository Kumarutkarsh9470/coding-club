1. Introduction

CampusPulse is a strategic data project designed to uncover factors influencing whether a student is in a romantic relationship, based on lifestyle, academic, and social survey data. This README guides you through the full pipeline, from raw data to explainable models.

2. Dataset Overview

Features: Demographics, academic performance (G1, G2, G3), lifestyle (Feature_1=Screen Time, Feature_2=Stress Level, Feature_3=Social Activity), alcohol use, family background, and more.

Target: romantic (yes/no)

3. Exploratory Data Analysis (EDA)

3.1 Data Cleaning & Missing Values

Identified columns with missing values.

Imputation strategies:

Numeric/ordinal: median

Categorical/binary: mode

Rule-based for Famsize using Pstatus
3.2 Visualizing Key Features

Histograms for Feature_1, Feature_2, Feature_3

Scatter plots to inspect correlations (e.g., screen time vs. grades)

Violin and box plots to compare distributions by romantic status

3.3 Insightful EDA Questions

Screen Time vs. Stress Level (Feature_1 vs Feature_2)

Stress vs. Health & Absences (violin plots)

Social Activity by Romantic Status (violin & count plots)

Parental Education vs. Grades (violin plots for Medu, Fedu)

Interaction of Screen Time, Social Activity, and Stress (bubble chart)

sns.violinplot(x='romantic', y='Feature_1', data=df)

...and more questions as explored.

4. Data Preprocessing

4.1 Imputation

Used SimpleImputer from scikit-learn for batch imputation.

Separate transformers for median and mode.

4.2 Encoding & Scaling

One-Hot Encoding for categorical features via OneHotEncoder

Standard Scaling for numeric features via StandardScaler

All steps combined in a ColumnTransformer and Pipeline:

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

5. Modeling

5.1 Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

5.2 Classifier Setup

Three balanced classifiers:

models = {
  'Logistic Regression': Pipeline([...]),
  'Random Forest':       Pipeline([...]),
  'SVM':                 Pipeline([...])
}

5.3 Performance Comparison

Evaluated accuracy, precision, recall, F1-score, ROC-AUC

Example:

print(classification_report(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_proba))

Results: Logistic and SVM ~0.61 AUC, Random Forest ~0.59 AUC

6. SHAP Explainability

6.1 Global Feature Importance

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_proc)[1]
shap.summary_plot(shap_values, X_train_proc, feature_names)

6.2 Local Explanations

idx_yes = np.where(rf_pipe.predict(X_test)==1)[0][0]
shap.plots.waterfall(explainer(X_test_proc[idx_yes]))

Interpret in plain language: which features push prediction toward "Yes" or "No".

7. How to Run

Install requirements: pip install -r requirements.txt (includes scikit-learn, pandas, shap, seaborn)

Place Dataset.csv in the project root.

Run the notebook task.ipynb or scripts:

python scripts/eda.py

python scripts/preprocess.py

python scripts/train_and_explain.py
