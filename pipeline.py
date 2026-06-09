# ============================================================
#  Healthcare Provider Fraud Detection — pipeline.py
#  Beginner-friendly, fully commented
# ============================================================

# STEP A: Import libraries
# pandas  = work with tables (like Excel in Python)
# numpy   = math operations
# sklearn = machine learning tools
# xgboost = powerful ML model for fraud detection
# matplotlib/seaborn = draw charts
# joblib  = save the trained model to disk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Healthcare Fraud Detection Pipeline")
print("=" * 60)

# ============================================================
# STEP 1: Load all 8 CSV files
# ============================================================
print("\n[1/8] Loading data files...")

DATA = "data/"          # folder where your CSVs live
OUT  = "outputs/"       # folder where results are saved
CHARTS = "charts/"      # folder where charts are saved

os.makedirs(OUT, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

# Training data (we know who is fraud here)
train_labels   = pd.read_csv(DATA + "Train-1542865627584.csv")
train_bene     = pd.read_csv(DATA + "Train_Beneficiarydata-1542865627584.csv")
train_inpat    = pd.read_csv(DATA + "Train_Inpatientdata-1542865627584.csv")
train_outpat   = pd.read_csv(DATA + "Train_Outpatientdata-1542865627584.csv")

# Unseen/Test data (we need to PREDICT fraud for these)
test_bene      = pd.read_csv(DATA + "Unseen_Beneficiarydata-1542969243754.csv")
test_inpat     = pd.read_csv(DATA + "Unseen_Inpatientdata-1542969243754.csv")
test_outpat    = pd.read_csv(DATA + "Unseen_Outpatientdata-1542969243754.csv")
test_providers = pd.read_csv(DATA + "Unseen-1542969243754.csv")

print(f"  Training providers  : {train_labels.shape[0]}")
print(f"  Fraud providers     : {(train_labels['PotentialFraud']=='Yes').sum()}")
print(f"  Inpatient claims    : {train_inpat.shape[0]}")
print(f"  Outpatient claims   : {train_outpat.shape[0]}")
print(f"  Unseen providers    : {test_providers.shape[0]}")

# ============================================================
# STEP 2: Feature Engineering for INPATIENT claims
# For each Provider, calculate summary statistics
# ============================================================
print("\n[2/8] Engineering inpatient features...")

def engineer_inpatient(df):
    # Convert date columns to proper date format
    df['AdmissionDt']  = pd.to_datetime(df['AdmissionDt'], errors='coerce')
    df['DischargeDt']  = pd.to_datetime(df['DischargeDt'], errors='coerce')
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], errors='coerce')
    df['ClaimEndDt']   = pd.to_datetime(df['ClaimEndDt'], errors='coerce')

    # How many days was the patient admitted?
    df['HospitalStayDays'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days
    # How long did the claim span?
    df['ClaimDurationDays'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days

    # Now group by Provider — one row per provider with summary stats
    grp = df.groupby('Provider').agg(
        IP_TotalClaims         = ('ClaimID', 'count'),
        IP_TotalReimbursed     = ('InscClaimAmtReimbursed', 'sum'),
        IP_AvgReimbursed       = ('InscClaimAmtReimbursed', 'mean'),
        IP_MaxReimbursed       = ('InscClaimAmtReimbursed', 'max'),
        IP_TotalDeductible     = ('DeductibleAmtPaid', 'sum'),
        IP_UniquePatients      = ('BeneID', 'nunique'),
        IP_UniqueAttendPhys    = ('AttendingPhysician', 'nunique'),
        IP_UniqueOperPhys      = ('OperatingPhysician', 'nunique'),
        IP_AvgHospitalStay     = ('HospitalStayDays', 'mean'),
        IP_AvgClaimDuration    = ('ClaimDurationDays', 'mean'),
    ).reset_index()
    return grp

train_ip_feat = engineer_inpatient(train_inpat)
test_ip_feat  = engineer_inpatient(test_inpat)
print(f"  Inpatient features created: {train_ip_feat.shape[1]-1} features for {train_ip_feat.shape[0]} providers")

# ============================================================
# STEP 3: Feature Engineering for OUTPATIENT claims
# ============================================================
print("\n[3/8] Engineering outpatient features...")

def engineer_outpatient(df):
    df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], errors='coerce')
    df['ClaimEndDt']   = pd.to_datetime(df['ClaimEndDt'], errors='coerce')
    df['ClaimDurationDays'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days

    grp = df.groupby('Provider').agg(
        OP_TotalClaims         = ('ClaimID', 'count'),
        OP_TotalReimbursed     = ('InscClaimAmtReimbursed', 'sum'),
        OP_AvgReimbursed       = ('InscClaimAmtReimbursed', 'mean'),
        OP_MaxReimbursed       = ('InscClaimAmtReimbursed', 'max'),
        OP_TotalDeductible     = ('DeductibleAmtPaid', 'sum'),
        OP_UniquePatients      = ('BeneID', 'nunique'),
        OP_UniqueAttendPhys    = ('AttendingPhysician', 'nunique'),
        OP_AvgClaimDuration    = ('ClaimDurationDays', 'mean'),
    ).reset_index()
    return grp

train_op_feat = engineer_outpatient(train_outpat)
test_op_feat  = engineer_outpatient(test_outpat)
print(f"  Outpatient features created: {train_op_feat.shape[1]-1} features for {train_op_feat.shape[0]} providers")

# ============================================================
# STEP 4: Feature Engineering for BENEFICIARY (patient) data
# ============================================================
print("\n[4/8] Engineering beneficiary features...")

def engineer_beneficiary(df):
    # Chronic condition columns (value 1=Yes, 2=No — we convert to 1/0)
    chronic_cols = [c for c in df.columns if c.startswith('ChronicCond_')]
    for col in chronic_cols:
        df[col] = (df[col] == 1).astype(int)

    # Total number of chronic conditions each patient has
    df['TotalChronicConds'] = df[chronic_cols].sum(axis=1)

    # Is the patient deceased? (DOD = Date of Death)
    df['IsDeceased'] = df['DOD'].notna().astype(int)

    # Age (approximate from DOB)
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['Age'] = (pd.to_datetime('2009-12-31') - df['DOB']).dt.days // 365

    # RenalDiseaseIndicator is 'Y' or 0 — convert to 1/0
    df['RenalDiseaseIndicator'] = (df['RenalDiseaseIndicator'] == 'Y').astype(int)

    # We don't have Provider in beneficiary data directly
    # So just return cleaned patient-level data
    return df

train_bene_clean = engineer_beneficiary(train_bene)
test_bene_clean  = engineer_beneficiary(test_bene)
print(f"  Beneficiary features ready: {train_bene_clean.shape[0]} patients")

# ============================================================
# STEP 5: Link beneficiary features to claims, then to Provider
# ============================================================
print("\n[5/8] Merging all data sources...")

# Fields we want from beneficiary data (patient-level averages per provider)
bene_summary_cols = ['BeneID', 'Age', 'TotalChronicConds', 'IsDeceased',
                     'RenalDiseaseIndicator', 'IPAnnualReimbursementAmt',
                     'OPAnnualReimbursementAmt', 'NoOfMonths_PartACov']

def merge_bene_to_claims(claims_df, bene_df):
    merged = claims_df.merge(
        bene_df[bene_summary_cols], on='BeneID', how='left'
    )
    return merged

# Merge beneficiary info into inpatient claims
train_ip_merged = merge_bene_to_claims(train_inpat, train_bene_clean)
test_ip_merged  = merge_bene_to_claims(test_inpat,  test_bene_clean)

# Aggregate beneficiary averages at Provider level from inpatient
def bene_provider_features(merged_df):
    grp = merged_df.groupby('Provider').agg(
        Bene_AvgAge              = ('Age', 'mean'),
        Bene_AvgChronicConds     = ('TotalChronicConds', 'mean'),
        Bene_PctDeceased         = ('IsDeceased', 'mean'),
        Bene_AvgIPReimbursement  = ('IPAnnualReimbursementAmt', 'mean'),
        Bene_AvgOPReimbursement  = ('OPAnnualReimbursementAmt', 'mean'),
    ).reset_index()
    return grp

train_bene_feat = bene_provider_features(train_ip_merged)
test_bene_feat  = bene_provider_features(test_ip_merged)

# Now merge all feature tables together on Provider
def build_provider_features(labels_or_providers, ip_feat, op_feat, bene_feat):
    df = labels_or_providers.copy()
    df = df.merge(ip_feat,   on='Provider', how='left')
    df = df.merge(op_feat,   on='Provider', how='left')
    df = df.merge(bene_feat, on='Provider', how='left')
    return df

train_final = build_provider_features(train_labels, train_ip_feat, train_op_feat, train_bene_feat)
test_final  = build_provider_features(test_providers, test_ip_feat, test_op_feat, test_bene_feat)

print(f"  Training dataset shape : {train_final.shape}")
print(f"  Test dataset shape     : {test_final.shape}")

# ============================================================
# STEP 6: Prepare data for ML model
# ============================================================
print("\n[6/8] Preparing data for model training...")

# Convert "Yes"/"No" fraud label to 1/0
train_final['FraudLabel'] = (train_final['PotentialFraud'] == 'Yes').astype(int)
print(f"  Fraud = 1 : {train_final['FraudLabel'].sum()} providers")
print(f"  Fraud = 0 : {(train_final['FraudLabel']==0).sum()} providers")

# Features = everything except Provider ID and the label columns
drop_cols = ['Provider', 'PotentialFraud', 'FraudLabel']
feature_cols = [c for c in train_final.columns if c not in drop_cols]

X = train_final[feature_cols].fillna(0)   # fill missing values with 0
y = train_final['FraudLabel']

X_test_final = test_final[feature_cols].fillna(0)

# Split training data: 80% to train the model, 20% to evaluate it
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples   : {len(X_train)}")
print(f"  Validation samples : {len(X_val)}")

# ============================================================
# STEP 7: Train the XGBoost Model
# ============================================================
print("\n[7/8] Training XGBoost model...")

# scale_pos_weight handles imbalanced data (only ~9% fraud cases)
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=fraud_ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

# Evaluate on validation set
y_pred       = model.predict(X_val)
y_prob       = model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f"\n  ✅ Model Results on Validation Set:")
print(f"     Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"     AUC Score : {auc:.4f}  (higher = better, max 1.0)")
print(f"\n  Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Not Fraud', 'Fraud']))

# ============================================================
# STEP 8: Save Charts
# ============================================================
print("\n[8/8] Saving charts and outputs...")

# Chart 1: Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=feature_cols)
top20    = feat_imp.nlargest(20)

plt.figure(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 20 Features for Fraud Detection', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(CHARTS + 'feature_importance.png', dpi=150)
plt.close()
print("  Saved: charts/feature_importance.png")

# Chart 2: Fraud vs Not Fraud distribution
plt.figure(figsize=(6, 4))
train_labels['PotentialFraud'].value_counts().plot(kind='bar', color=['steelblue','crimson'])
plt.title('Fraud vs Legitimate Providers in Training Data')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CHARTS + 'fraud_distribution.png', dpi=150)
plt.close()
print("  Saved: charts/fraud_distribution.png")

# Chart 3: Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(CHARTS + 'confusion_matrix.png', dpi=150)
plt.close()
print("  Saved: charts/confusion_matrix.png")

# Save trained model
joblib.dump(model, OUT + 'fraud_model.pkl')
print("  Saved: outputs/fraud_model.pkl")

# ============================================================
# GENERATE SUBMISSION FILE
# ============================================================
test_prob  = model.predict_proba(X_test_final)[:, 1]
test_pred  = model.predict(X_test_final)

submission = pd.DataFrame({
    'Provider'      : test_final['Provider'],
    'Probability'   : test_prob.round(4),
    'PredictedClass': ['Yes' if p == 1 else 'No' for p in test_pred]
})

submission.to_csv(OUT + 'Submission.csv', index=False)
print(f"\n  ✅ Saved: outputs/Submission.csv  ({len(submission)} rows)")
print(f"     Predicted Fraud     : {(submission['PredictedClass']=='Yes').sum()}")
print(f"     Predicted Not Fraud : {(submission['PredictedClass']=='No').sum()}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE!")
print("  Check your outputs/ and charts/ folders.")
print("=" * 60)
