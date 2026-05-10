# Autism Spectrum Disorder Prediction
## Machine Learning Analysis — Industrial Training Project
**Company:** HMIES Solutions  
**Student:** Diploma in Computer Engineering (Final Year)  
**Semester:** 6th (Industrial Training)

---

## 1. Project Objective
To predict Autism Spectrum Disorder (ASD) using machine learning classification algorithms on two real-world datasets — child autism screening data and general autism screening data.

---

## 2. Datasets Used

### Dataset 1: `Autism_Child_data_in_excel.xlsx`
- **Records:** 292 children
- **Features:** 21 columns
- **Target Column:** `Class` (YES / NO)
- **Key Features:** Age, Gender, Ethnicity, Jaundice, Family Autism History, AQ-10 Scores (A1–A10), Result

### Dataset 2: `autism_screening.csv`
- **Records:** 704 individuals
- **Features:** 21 columns
- **Target Column:** `Class/ASD` (YES / NO)
- **Key Features:** Same structure as above

---

## 3. Data Preprocessing
1. Removed leading/trailing whitespace from string columns
2. Label Encoded all categorical columns (gender, ethnicity, etc.)
3. Filled 2 missing age values in CSV with median
4. Split data: **80% training / 20% testing**

---

## 4. Five Machine Learning Algorithms Applied

### Algorithm 1: Logistic Regression
- **Type:** Classification
- **Accuracy:** 100.00%
- **How it works:** Calculates the probability of autism using a sigmoid (S-shaped) function. If probability > 0.5 → YES (Autism)

### Algorithm 2: Decision Tree
- **Type:** Classification
- **Accuracy:** 100.00%
- **How it works:** Builds a tree of yes/no questions based on features. Each leaf gives a final prediction.

### Algorithm 3: Random Forest
- **Type:** Ensemble Classification
- **Accuracy:** 100.00%
- **How it works:** Combines 100 decision trees and takes the majority vote. More robust than a single tree.

### Algorithm 4: Support Vector Machine (SVM)
- **Type:** Classification
- **Accuracy:** 83.05%
- **How it works:** Finds the best hyperplane (boundary) that separates autism vs non-autism cases.

### Algorithm 5: Naive Bayes
- **Type:** Probabilistic Classification
- **Accuracy:** 96.61%
- **How it works:** Uses Bayes' Theorem to calculate the probability of autism given the feature values.

---

## 5. Results Summary

| Algorithm           | Accuracy |
|---------------------|----------|
| Logistic Regression | 100.00%  |
| Decision Tree       | 100.00%  |
| Random Forest       | 100.00%  |
| Naive Bayes         | 96.61%   |
| SVM                 | 83.05%   |

**Best Performing Algorithms:** Logistic Regression, Decision Tree, and Random Forest all achieved 100% accuracy on the test set.

---

## 6. Visualizations Created (10 Charts)
1. Bar chart — Autism Class Distribution (YES vs NO)
2. Pie chart — Proportion of autism cases
3. Histogram — Age distribution by class
4. Grouped bar — Gender vs Autism
5. Grouped bar — Average AQ-10 scores per question
6. Heatmap — Feature correlation matrix
7. Scatter plot — Age vs Total Score
8. Bar charts — Jaundice & Family history impact
9. Horizontal bar — Ethnicity distribution
10. Line graph — Average score across ages

---

## 7. Dashboard (Streamlit)
Interactive 5-page dashboard:
- **Page 1:** Overview & EDA
- **Page 2:** 10 Data Visualizations
- **Page 3:** ML Models — Confusion Matrix, ROC Curve, Feature Importance
- **Page 4:** Model Comparison — Radar Chart, ROC overlay, Summary Table
- **Page 5:** Real-time Autism Prediction for a new child

### To Run the Dashboard:
```bash
cd autism_project
streamlit run app.py
```

---

## 8. Libraries Used
| Library | Purpose |
|---------|---------|
| pandas | Data loading & manipulation |
| numpy | Numerical computation |
| matplotlib / seaborn | Static visualizations |
| plotly | Interactive charts |
| scikit-learn | Machine learning algorithms |
| streamlit | Dashboard / web app |
| openpyxl | Reading Excel files |

---

## 9. Conclusion
- The AQ-10 screening questions (A1–A10) are the most important features for autism prediction.
- Children with a score ≥ 6 are likely to be autistic (clinically validated threshold).
- Machine learning can significantly assist in early autism detection.
- Random Forest and Logistic Regression are recommended for deployment.

---

*Project completed as part of 6th Semester Industrial Training at HMIES Solutions.*
