# 💳 Credit Risk Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the credit risk of loan applicants using machine learning techniques.  
The system classifies applicants as **GOOD (creditworthy)** or **BAD (high risk)** based on demographic and financial attributes, helping banks make informed lending decisions.

The final model is deployed as an interactive **Streamlit web application**.

---

## ❓ Problem Statement
Financial institutions face significant losses when loans are approved for high-risk applicants.  
The challenge is to accurately assess an applicant’s creditworthiness before loan approval to minimize default risk.

---

## 🎯 Business Objective
- Reduce loan default rates  
- Identify high-risk applicants early  
- Support data-driven lending decisions  
- Improve overall credit portfolio quality  

---

## 🤖 Machine Learning Objective
- Perform Exploratory Data Analysis (EDA)  
- Preprocess and encode customer data  
- Train and compare multiple classification models  
- Select the best model using business-relevant metrics  
- Deploy the final model using Streamlit  

---

## 📊 Dataset Description
- **Dataset**: German Credit Risk Dataset  
- **Total Records**: 1000  
- **Target Variable**: `Risk` (GOOD / BAD)

### Feature Types
- **Numerical**: Age, Credit Amount, Duration  
- **Categorical**: Sex, Job, Housing, Saving Accounts, Checking Account, Purpose  

---

## 🧹 Data Preprocessing
- Handled missing values using `fillna()`  
- Removed irrelevant index column  
- Encoded categorical variables using **Label Encoding**  
- Used **stratified train-test split** to preserve class balance  

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Credit amount and loan duration are strong indicators of credit risk  
- Higher loan amounts and longer durations increase default probability  
- Applicants with low savings and checking balances show higher risk  
- Financial behavior features are more influential than demographic attributes  

---

## 🧠 Models Trained
- Decision Tree  
- Random Forest  
- Extra Trees Classifier  
- XGBoost Classifier  

---

## 📈 Model Evaluation
Models were evaluated using:
- Accuracy  
- Confusion Matrix  
- Classification Report  
- Cross-Validation  
- **ROC-AUC (Primary Metric)**  

**Why ROC-AUC?**  
Credit risk prediction is a probability-ranking problem, and ROC-AUC measures how well the model separates good and bad applicants across thresholds.

---

## 🏆 Final Model Selection
Although Random Forest achieved slightly higher accuracy, **XGBoost achieved the highest ROC-AUC score**, indicating superior separation between good and bad credit risks.

- **Final Model**: XGBoost  
- **Saved Model**: `XGB_Credit_model.pkl`

---

## 🖥️ Streamlit Application Preview

<img src="https://github.com/user-attachments/assets/032f315d-312d-4d13-bd57-b6ac338b28e3" width="100%" />

### 📌 Application Overview
The screenshot above shows the deployed **Credit Risk Prediction Streamlit application**.  
Users can enter applicant details such as age, job level, housing type, savings, credit amount, and loan duration.

### 🔍 Prediction Logic
- The application uses a trained **XGBoost model**
- Predicts whether an applicant is:
  - **GOOD** – Creditworthy  
  - **BAD** – High risk  
- Displays a **probability score** indicating model confidence

### ⚖️ Risk Sensitivity Control
The prediction threshold allows financial institutions to adjust approval strictness based on business risk appetite.

---

## 🚀 Deployment – Streamlit Application
The trained XGBoost model is deployed using Streamlit and provides:
- User-friendly input form  
- Real-time credit risk prediction  
- Probability-based confidence scores  
- Business-aligned decision support  

---
## 🗂 Project Structure

German-Credit-Risk-Machine-Learning/
│
├── Credit Risk Analysis model.ipynb
├── credit_risk_app.py
├── XGB_Credit_model.pkl
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── target_encoder.pkl
├── requirements.txt
└── README.md
---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App
```bash
python -m streamlit run credit_risk_app.py
```
### 3️⃣ Open in Browser
http://localhost:8501

## ✅ Key Takeaways
- ROC-AUC is more reliable than accuracy for credit risk prediction problems  
- XGBoost performs exceptionally well on structured financial datasets  
- Probability-based predictions enable better and safer lending decisions  

---

## 🔮 Future Enhancements
- SHAP-based model explainability for transparent predictions  
- Threshold optimization based on bank risk appetite  
- Model calibration to improve probability estimates  
- Integration with real-world banking or loan management systems  

---

## 👩‍💻 Author
**Gouthami Gottimukkala**  
Aspiring Data Analyst | Data Scientist  
Passionate about Machine Learning, Data Analytics, and Building Real-World Data-Driven Applications
