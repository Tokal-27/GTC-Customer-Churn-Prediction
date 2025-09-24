# 📊 Telco Customer Churn Prediction

This project predicts **customer churn** (whether a customer will leave or stay) using the famous **Telco Customer Churn dataset**.  
We perform **data preprocessing, feature engineering, and machine learning**, then deploy the final model as an **interactive Streamlit web app**.

---

## 📁 Project Structure
├── Telco_Customer_Churn.csv # Dataset
├── churn_prediction.ipynb # Main Colab notebook (EDA, modeling)
├── app.py # Streamlit app
├── requirements.txt # Dependencies
├── README.md # Project documentation


## ⚙️ Features
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering (dummy encoding, scaling, outlier removal)  
- Train/Test split  
- Class balancing with **SMOTE**  
- Model training & evaluation with:
  - Random Forest (GridSearchCV tuned)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM ✅ final model)
  - Gradient Boosting
  - XGBoost  
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC
  - Confusion Matrix
  - ROC Curves comparison  
- **Streamlit deployment** for live churn prediction  


## 📊 Dataset
- **Source:** Telco Customer Churn dataset  
- **Size:** ~7043 rows × 21 columns  
- **Target Variable:** `Churn` (Yes/No)  
- **Imbalance:** ~26% churned vs. 74% retained  


## 🛠️ Installation & Setup
### 1. Clone the repo
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
(requirements.txt example)

nginx
Copy code
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
xgboost
streamlit
3. Run the notebook (optional for training)
bash
Copy code
jupyter notebook churn_prediction.ipynb
4. Run the Streamlit app
bash
Copy code
streamlit run app.py
📈 Results
Final selected model: Support Vector Classifier (SVC)

Key Performance (example results – update with your run):

Model	Accuracy	Precision	Recall	F1	ROC-AUC
SVC (final)	0.82	0.67	0.64	0.65	0.87
Random Forest	0.84	0.72	0.66	0.69	0.89
Logistic Regression	0.81	0.65	0.61	0.63	0.86
XGBoost	0.83	0.71	0.65	0.68	0.88


🚀 Deployment
The app is deployed using Streamlit:

Input customer details through the UI

Get real-time churn prediction ("Yes" / "No")

Example UI:

yaml
Copy code
Gender: Female
Tenure: 12 months
MonthlyCharges: 70.5
...
👉 Prediction: Customer will Churn ✅
📌 Future Improvements
Deploy to Streamlit Cloud / Hugging Face Spaces / AWS for public access.

Integrate SHAP / LIME for explainability.

Explore Deep Learning models (PyTorch / Keras).
