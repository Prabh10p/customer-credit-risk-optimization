# Credit Risk–Based Loan Approval System
![Credit Risk Modeling](https://miro.medium.com/v2/resize:fit:1400/1*zy6cELd7h8yxT9XA3sbJvw.png) 

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen?logo=streamlit)](https://loan-default-predictor.streamlit.app)  



## 📌 **Table of Contents**
1. [Overview](#Overview)
2. [Key Features](#key-features)
3. [Tech Stack](#tech-stack)
4. [Dataset](#dataset)
5. [Project Workflow](#project-workflow)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Explainability: SHAP & LIME](#explainability-shap--lime)
9. [Run Project Locally](#)
10. [Future Improvements](#future-improvements)
11. [Author](#Author)



# 1 📌 Overview
This project is an end-to-end credit risk prediction pipeline that uses machine learning to assess whether a loan applicant is likely to default.
It combines data preprocessing, EDA, feature engineering, model training, explainable AI (SHAP & LIME), and a Streamlit web app for deployment.
The core model achieved ~85% accuracy using XGBoost Classifier with optimized hyperparameters.


# 2 🚀 Key Features
- **Data Preprocessing**: Missing value handling, encoding, scaling.
- **EDA Visualizations**: Interactive graphs for data insights.
- **Machine Learning Models**:
  - Logistic Regression
  - XGBoost Classifier
- **Business Profit Simulation**: Threshold tuning for optimal approval rate.
- **Explainable AI**:
  - Global & Local feature importance using SHAP & LIME.
- **Deployment**:
  - Streamlit Cloud dashboard with model integration.


# 3 ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-lightblue?logo=plotly)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-lightgreen?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-f7931e?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/SHAP-Model%20Explainability-red)
![LIME](https://img.shields.io/badge/LIME-Local%20Interpretability-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)
![Git](https://img.shields.io/badge/Git-Version%20Control-orange?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)
![venv](https://img.shields.io/badge/venv-Environment%20Management-blue)
![conda](https://img.shields.io/badge/conda-Environment%20Management-green?logo=anaconda)


# 4 📂 Dataset
- Source: Kaggle Credit Risk Dataset (or your dataset source)
- Rows: ~32,000
- Columns: 20+ features, including:
  - person_age — Age of applicant
  - person_income — Annual income
  - loan_amount — Requested loan amount
  - loan_percent_income — Loan amount as a percentage of income
  - credit_history_duration — Length of applicant’s credit history
  - loan_intent — Purpose of loan (e.g., EDUCATION, PERSONAL, VENTURE)
  - loan_status — Target variable: 1 for default, 0 for no default



# 5 🔄 Project Workflow
## 1. Data Collection
- Load dataset from source and save into /Artifacts directory.

## 2. src/Components — Core Data Processing Modules
- data_ingestion.py → Loads raw dataset, splits into train & test sets, and stores them in /Artifacts.
- data_transformation.py → Handles missing values, encodes categorical features, and scales numerical features.
- model_trainer.py → Trains models (Logistic Regression, XGBoost) and performs hyperparameter tuning.
## 3. src/Pipeline — Model Deployment Pipeline
- Model_pipeline.py →Loads saved preprocessor and trained model.
- Applies preprocessing and prediction steps to new incoming data.
## 4. Utility & Support Files in src
- utils.py → Save/load models and preprocessing objects.
- Exception.py → Centralized custom exception handling.
- logger.py → Application-wide logging setup.
## 5. Notebooks (/notebooks) — Experimentation & Analysis
- credit_risk_EDA.ipynb → Exploratory Data Analysis & Visualization.
- feature_engineering.ipynb → Creation and selection of meaningful features.
- model_explainability.ipynb → SHAP & LIME-based interpretation of predictions.
## 6. Model Evaluation
- Compare trained models using Accuracy, F1 Score, and ROC AUC.
## 7. Model Explainability
- Global Interpretability: SHAP to understand feature importance across all predictions.
- Local Interpretability: LIME to explain predictions for specific customer cases.
## 8. Deployment & Streamlit Dashboard
- Preprocessor & model are saved as .pkl files for production.
- app.py → Interactive Streamlit UI for loan default prediction.





# 6. 📊 Exploratory Data Analysis
- The Exploratory Data Analysis (EDA) phase was crucial in understanding the dataset’s structure, relationships, and anomalies before moving into modeling.
- Key steps performed:
   - Missing Value Analysis: Identified and imputed missing values where applicable.
   - Outlier Detection: Detected anomalies in continuous variables like loan_amount and annual_income using boxplots and Z-score.
   - Distribution Checks: Visualized distributions of numeric and categorical features to detect skewness.
   - Correlation Heatmaps: Measured relationships between variables, highlighting potential multicollinearity.
   - Target Variable Relationship: Compared default rates across features like employment_length, credit_history_length, and loan_purpose.


# 7. 🤖 Model Training & Evaluation
- We trained multiple ML models to identify the best-performing one for predicting loan default risk.
Models Tested

- Logistic Regression (baseline)
- XGBoost Classifier (boosted tree-based model with high accuracy potential)
- Approach:
  - Train-Test Split (80/20) to ensure unbiased evaluation.
  - Cross-Validation for robust performance measurement.
  - Hyperparameter Tuning with GridSearchCV for optimal model parameters.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC score.
| Model               | Accuracy | F1 Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 0.70     | 0.70     | 0.8741  |
| XGBoost Classifier  | 0.85     | 0.83     | 0.87    |


# 8📊 Model Interpretability
To ensure transparency and trust in our predictions, we used SHAP and LIME to interpret model behavior.

## SHAP Summary Plot

<p align="center">
  <img src="Images/Screenshot%202025-08-09%20at%204.09.25%E2%80%AFPM.png" alt="SHAP Summary Plot" width="700"/>
  <br>
  <em>SHAP summary plot showing feature impact on loan default prediction.</em>
</p>


From the SHAP summary plot:
- Loan Amount and Person Income are the most influential features in predicting credit risk.
- Higher Loan Amount and higher Loan Percent Income tend to increase default risk.
- Employment Duration and Person Age also play significant roles, with lower values often linked to higher risk.
- Certain categorical features like Home Ownership Type = RENT and Credit History Duration also have notable effects.
- Loan intent categories such as VENTURE, EDUCATION, and PERSONAL influence predictions differently depending on the context.

## LIME Local Explanation
<p align="center">
  <img src="Images/Screenshot%202025-08-11%20at%206.21.43%E2%80%AFPM.png" alt="LIME Local Explanation" width="700"/>
  <br>
  <em>LIME local explanation for a single customer instance.</em>
</p>

From the LIME explanation:

- For the specific customer examined, High Loan Amount and Low Person Income strongly increased the probability of default.
- A short Credit History Duration further contributed to higher predicted risk.
- Factors like Employment Duration and certain loan purposes reduced the risk slightly, but not enough to change the classification outcome.

# 9 🚀 Run Project Locally
Follow these steps to set up and run the project on your local machine:
## 1️⃣ Clone the Repository
- git clone https://github.com/your-username/credit_risk_prediction.git
- cd credit_risk_prediction
## 2️⃣ Create Virtual Environment
### For Windows
- python -m venv venv
- venv\Scripts\activate

#### For macOS / Linux
- python3 -m venv venv
- source venv/bin/activate
## 3️⃣ Install Dependencies
- pip install -r requirements.txt
## 4️⃣ Run the Streamlit App
- streamlit run app.py
- Open the link shown in the terminal (usually http://localhost:8501) to interact with the dashboard.


# 10 📌 Future Enhancements

- Deploy to Cloud (AWS, GCP, Heroku) for global access
- Add more algorithms (CatBoost, LightGBM) for comparison
- Automated data pipeline for real-time loan application scoring
- Integration with APIs for fetching live customer credit data
- Advanced Explainability with interactive dashboards
- User authentication for secure access

# 11 👨‍💻 Author

**Prabhjot Singh**  
🎓 B.S. in Information Technology, Marymount University  
🔗 [LinkedIn](https://www.linkedin.com/in/prabhjot-singh-10a88b315/)  
🧠 Passionate about data-driven decision making, analytics, and automatio

