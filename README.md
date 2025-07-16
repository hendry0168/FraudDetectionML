# Predicting Rare Events with Decision Tree, Random Forest, and XGBoost

This project explores and compares multiple machine learning models to detect rare positive events (e.g. fraud). The models were trained on imbalanced data, and later improved using SMOTE to generate synthetic minority samples.

---

## 🔍 Models Evaluated

| Model          | Accuracy | Precision | Recall | F1-score |
|----------------|----------|-----------|--------|----------|
| Decision Tree  | 0.99909  | 0.67262   | 0.83088| 0.74342  |
| Random Forest  | 0.99960  | 0.93220   | 0.80882| 0.86614  |
| XGBoost        | 0.99961  | 0.93277   | 0.81618| 0.87059  |

---

## 🧪 After Applying SMOTE (Data Balancing)

| Model          | Accuracy | Precision | Recall | F1-score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.99988  | 0.99977   | 1.00000| 0.99988  |
| XGBoost        | 0.99966  | 0.99932   | 1.00000| 0.99966  |

---

## 🧠 Key Learnings

- SMOTE greatly improved the model’s ability to detect the minority class (positive cases).
- Random Forest outperformed XGBoost slightly after SMOTE, likely due to its simplicity and robustness to synthetic data.
- Even with high accuracy before SMOTE, precision and recall were much lower — showing how important it is to go beyond accuracy.

---

## 📚 Techniques Used

- Python (Pandas, scikit-learn, imbalanced-learn, XGBoost)
- SMOTE for resampling imbalanced data
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

---

## 🔄 How to Reproduce
1. Clone this repo  
2. Run `credit_card_fraud.py` (or Jupyter notebook)  

