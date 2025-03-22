# 🫀 Cardiovascular Disease Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen) 
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-blue) 
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## 🚀 Overview
This project predicts the **risk of cardiovascular disease** based on patient health indicators. It is **deployed on Streamlit Cloud**, allowing users to enter personal health data and receive an instant prediction along with feature impact analysis using **SHAP (SHapley Additive Explanations)**.

🔗 **Live App**: [Click here to try it](https://heart-attack-analysis-predict.streamlit.app/)

The data is retrieved from a Kaggle Competition: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

---

## 🏆 Features
✅ **BMI Calculator** – Quickly calculate your Body Mass Index (BMI).  
✅ **Cardiovascular Risk Prediction** – Predicts the likelihood of having heart disease.  
✅ **SHAP Interpretability** – Explains **why** the model made its prediction.  
✅ **Feature Importance Analysis** – Displays which health factors contribute most to the risk.  
✅ **Model Performance Evaluation** – Shows accuracy and classification report.  

---

## 📂 Project Structure

```
heart-attact-analysis/
│
├── data/
│   ├── cardio_train.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│
├── models/
│   ├── best_model_rf.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_explainability.py
│   ├── model_training.py
│
├── app.py
├── run_pipeline.sh
├── requirements.txt
├── README.md
```
---

## 🧭 App Features & Tabs

The app is organized into three intuitive tabs:

### 🔹 1. **BMI Calculator**
- A simple tool to calculate **Body Mass Index (BMI)**.
- Users input their **height (cm)** and **weight (kg)**.
- The app returns their **BMI value**, which is a helpful indicator of obesity—a key risk factor for heart disease.

---

### 🔹 2. **Heart Risk Prediction**
- Users enter health information including:
  - **Age**, **Gender**
  - **Systolic & Diastolic Blood Pressure**
  - **Cholesterol & Glucose levels**
  - **Smoking**, **Alcohol Consumption**, **Physical Activity**
  - **BMI** (can be copied from Tab 1)
- The model predicts if you're at **risk (1)** or **not at risk (0)** of heart disease.
- Uses **SHAP** to visualize how each input contributed to the prediction.
  - Shows both a **summary bar chart** of feature impact and a **waterfall chart** of the current prediction.

---

### 🔹 3. **Model Performance**
- Evaluates the trained model on test data (`X_test`, `y_test`).
- Displays:
  - 📊 **Classification Report**: Precision, Recall, F1-score, Support for each class.
  - ✅ **Overall Accuracy** as a Streamlit metric.
- Helps ensure the model is reliable and not overfitting.

---
## 💡 How It Works

1. **Model Training**
   - Trained using a **RandomForestClassifier**.
   - Hyperparameters tuned via **GridSearchCV**.
   - Evaluated with cross-validation and tested on a holdout set.

2. **Model Explainability**
   - **SHAP (SHapley Additive Explanations)** shows which features influenced predictions the most.
   - Feature contributions are visualized for individual predictions.

3. **Deployment**
   - App is deployed on **Streamlit Cloud** and is live.

---

## 📈 Model Results

The model used in this project is a **Random Forest Classifier**, optimized using **GridSearchCV** with 5-fold cross-validation. Below are the final training results and selected hyperparameters.

---

### ✅ Best Model Parameters

```json
{
  "bootstrap": true,
  "max_depth": 10,
  "min_samples_leaf": 4,
  "min_samples_split": 2,
  "n_estimators": 200
}
```
These parameters provided the best balance between model complexity and generalization performance.

---

### 📝 Model Performance on Training Data

| **Metric**                    | **Value** |
|------------------------------|-----------|
| Best Cross-Validation Score  | **0.7351** |
| Training Accuracy            | **0.7488** |

The small gap between CV and training accuracy suggests the model generalizes well and avoids overfitting.

### 📊 Classification Report (Training Set)

| **Class**         | **Precision** | **Recall** | **F1-Score** | **Support** |
|------------------|---------------|------------|--------------|-------------|
| 0 (No Disease)   | 0.72          | 0.81       | 0.76         | 28,037      |
| 1 (Disease)      | 0.78          | 0.69       | 0.73         | 27,963      |
| **Accuracy**     | -             | -          | **0.75**     | 56,000      |
| **Macro Avg**    | 0.75          | 0.75       | 0.75         | 56,000      |
| **Weighted Avg** | 0.75          | 0.75       | 0.75         | 56,000      |

### 🔍 Interpretation

- **Precision** is slightly higher for class `1` (Disease), indicating fewer false positives.
- **Recall** is stronger for class `0` (No Disease), meaning healthy individuals are more likely to be correctly identified.
- **F1-scores** for both classes are balanced, indicating good overall performance.

💡 This balance is crucial in a medical context where both **false positives** and **false negatives** carry consequences.

