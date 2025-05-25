#Task 1: Relationship Prediction Analysis from Student Data

This project explores student behavioral and academic data to analyze the likelihood of being in a romantic relationship. The analysis involves end-to-end steps including EDA, data preprocessing, classification modeling, performance evaluation, and explainability through SHAP.

#LEVEL 1: Exploratory Data Analysis (EDA)

#Objectives:
- Understand hidden/unknown features.
- Explore relationships between behavioral indicators and academic or social variables.

#Findings:
- Feature 1: Likely represents weekly screen time. Correlates positively with internet access and failures.
- Feature 2: Represents stress levels. Shows non-linear effect on grades—grades rise with stress up to a point, then fall.
- Feature 3: Likely social activity. Correlates positively with going out and affects romantic relationships.

#Visual Analysis:
- Correlation heatmaps, scatter plots, and histograms were used to identify trends.
- Screen time and social activity interplay showed visible influence on academic performance and social behavior.


#LEVEL 2: Handling Missing Values

- Categorical Scales (e.g., famsize, traveltime): Imputed using 'most_frequent'.
- Continuous Variables (e.g., Feature_1): Imputed using 'median'.

 #LEVEL 3: Insightful Questions and Answers

1. Does social activity differ by romantic status or family relationship? 
    No significant difference; possibly due to couple activities being counted as social.

2. Is there a link between screen time, social activity, and stress?  
    Higher social activity = lower stress. Higher screen time = lower social activity.

3. Do students with low parental education show higher stress/screen time? 
    No clear stress trend, but lower parental education corresponds to higher screen time.

4. What’s the trade-off between stress and social activity among students with high absences?  
    High absence students tend to have moderate-to-low social activity and varying stress.

5. Does travel time affect grades and absences?
    Longer travel time correlates with lower grades but not necessarily higher absences.

6. Does stress impact absenteeism?  
    Yes. Higher stress levels correspond to more absences, especially at moderate stress.

7. Are students with educated parents more likely to be in relationships?  
    Not conclusively. Students with very low Medu (mother's education) are rarely in relationships.

# LEVEL 4: Classification Models

# Logistic Regression
- Accuracy: 61%
- F1-Score (Romantic=Yes): 0.51  
- Confusion Matrix:
  - True Positives: 40
  - False Positives: 45
  - True Negatives: 78
  - False Negatives: 32

# Random Forest
- Accuracy: 63%
- F1-Score (Romantic=Yes): 0.23  
- Performs well for predicting singles but poorly for those in relationships.

# SVM
- Similar performance to Logistic Regression.
- Decision boundaries are more linear.

# Top 10 Important Features (Random Forest):
| Rank | Feature         | Importance |
|------|------------------|------------|
| 1    | G3 (Final Grade) | 0.0674     |
| 2    | G2               | 0.0638     |
| 3    | G1               | 0.0634     |
| 4    | Absences         | 0.0546     |
| 5    | Feature_1        | 0.0536     |
| ...  | ...              | ...        |


# LEVEL 5: Model Explainability with SHAP

#Decision Boundaries (Goout vs Walc):
- Higher weekend alcohol use and moderate going-out frequency → More likely in romantic relationships.
- Lower alcohol use and social activity → More likely single.
- Boundaries are non-linear, indicating interaction effects.

#SHAP Summary:
- Most feature interactions are minor (near 0 SHA interaction value).
- Few combinations show high influence on the prediction, suggesting local interactions matter more than global ones.

#Waterfall Plots:
- SHAP waterfall plots visualize the additive impact of each feature for individual predictions.
- Some features (e.g., Feature_15, Feature_21) have consistent negative effects on the romantic prediction.

#Conclusion

- A student's screen time, stress, social activity, and grades collectively influence their romantic status prediction.
- The best model performance (~63% accuracy) is from **Random Forest**, though it struggles to correctly predict those in relationships.
- SHAP helps uncover both global and local feature contributions, revealing subtle interactions beyond simple correlation.


#Tools Used
- Python, Pandas, Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest, SVM)
- SHAP for interpretability




