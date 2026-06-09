# Healthcare Provider Fraud Detection

## Project Overview
Predicting potentially fraudulent healthcare providers 
based on Medicare insurance claims data using Machine Learning.

## Model Results
- Accuracy  : 93.3%
- AUC Score : 0.9531
- Fraudulent Providers Detected : 144 out of 1,353

## Live Streamlit Dashboard
https://fraud-detection-jzfpff5dptinxgwhgbfvkj.streamlit.app/

## Project Files
| File | Description |
|------|-------------|
| pipeline.py | Main ML pipeline — loads data, trains model, saves predictions |
| app.py | Streamlit dashboard |
| requirements.txt | Python libraries needed |
| outputs/Submission.csv | Final fraud predictions |
| outputs/FraudDetection_Notebook.html | Complete analysis notebook |
| Analysis_Report.docx | Approach, findings and recommendations |

## Tech Stack
- Python 3.11
- XGBoost
- Scikit-learn
- Streamlit
- Pandas, NumPy, Matplotlib, Seaborn

## Dataset
Medicare claims data — Inpatient, Outpatient and Beneficiary records
- Training : 5,410 providers
- Test/Unseen : 1,353 providers
