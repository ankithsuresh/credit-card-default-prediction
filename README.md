# Credit Card Default Prediction (UCI Taiwanese Dataset)

This project predicts **credit card default** using the well-known **“Default of Credit Card Clients”** dataset (30,000 clients, Taiwan, 2005). The analysis is implemented in R and walks through data cleaning, feature engineering, and model comparison across logistic regression and machine learning classifiers.

The focus is on:
- Using **behavioral and financial features** to predict default
- Handling **class imbalance** (default vs. non-default)
- Comparing **logistic regression** with **Random Forest, SVM, and KNN**

---

## Dataset

- **Name:** Default of Credit Card Clients  
- **Source (UCI):** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
- **Original paper:**  
Yeh, I.-C., & Lien, C.-H. (2009). *The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients.* Expert Systems with Applications, 36(2), 2473–2480.

> The original `.xls` file is **not** included in this repo.  
> To run the code, download it from UCI and place it in the `data/` folder.

---

## Methodology (High Level)

1. **Preprocessing & Cleaning**
   - Load the original `.xls` dataset.
   - Remove non-predictive `ID`.
   - Recode education categories and repayment status (e.g., map all non-positive delays to “no delay”).
   - Exclude sensitive demographic fields (e.g., gender, marital status) to focus on **behavioral** signals.

2. **Baseline Models**
   - Fit **logistic** and **probit** regression models.
   - Assess performance using ROC–AUC and confusion matrices.
   - Use the logistic model as the main baseline.

3. **Feature Engineering**
   - Create aggregate and trend features, such as:
     - Average and maximum delays across months  
     - Average bill amount and payment amount  
     - Utilization-like ratios

4. **Machine Learning Models**
   - Apply simple **oversampling** of defaulters in the training set to handle class imbalance.
   - Train and compare:
     - **Random Forest**
     - **Support Vector Machine (radial kernel)**
     - **K-Nearest Neighbors (KNN)**
   - Evaluate on a held-out test set using **Accuracy, Precision, Recall/Sensitivity, F1-score, and AUC**.

In this implementation, **Random Forest** typically achieves the best trade-off, improving recall for defaulters compared with the baseline logistic model while keeping precision reasonable.

---
                      
