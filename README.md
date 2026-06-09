# Loan Default Risk Prediction 

## Project Overview
This project applies a Random Forest classifier to predict loan defaults using historical financial data. By identifying high-risk borrowers before approval, this model helps financial institutions minimize credit losses while balancing profitability and risk.

## Key Results & Business Impact
* **High Discriminative Power:** Achieved an ROC-AUC score of 0.8654.
* **Optimized for Risk (85% Recall):** Lowered the decision threshold to 0.40 to successfully identify 85% of actual defaulters. This trade-off prioritizes catching high-risk loans over minimizing false positives.
* **Data-Driven Insights:** Revealed that Credit Utilization and 90-Day Late Payments are the most critical predictors of default.

## Dataset & Tech Stack
* **Data:** 150,000 borrower records from the Kaggle "Give Me Some Credit" dataset. The dataset exhibits severe class imbalance with a ~7% default rate.
* **Tech:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn.
* **Model:** Random Forest Classifier optimized via `RandomizedSearchCV`.

## Methodology
1. **Preprocessing:** Handled missing values via median imputation strictly *after* train-test splitting (80/20) to prevent data leakage.
2. **Class Imbalance:** Applied `class_weight='balanced'` to heavily penalize minority class (default) misclassifications.
3. **Tuning:** Optimized tree depth, estimators, and split criteria using `RandomizedSearchCV`.
4. **Calibration:** Calibrated the probability threshold down to 0.40 to prioritize Recall and align with business risk goals.

## Future Improvements
* Benchmark against Logistic Regression and XGBoost models.
* Integrate SHAP (SHapley Additive exPlanations) for advanced model explainability.
* Deploy as an interactive Streamlit web application.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the `cs-training.csv` dataset from Kaggle and place it in the root directory.
4. Run the Jupyter notebook or Python script to train the model and generate visualizations.

---
*Author: Adhithya S*
