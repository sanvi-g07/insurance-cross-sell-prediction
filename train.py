import kagglehub

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

OUTPUT_FILE = 'insurance_model.bin'

NUM_COLS = ['Age', 'Annual_Premium', 'Vintage', 'Annual_Premium_log', 'Premium_per_Vintage']
CAT_COLS = [
    'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Damaged_and_Uninsured',
]
TARGET_COL = 'Response'
VEHICLE_AGE_ORDER = ['< 1 Year', '1-2 Year', '> 2 Years']

BEST_PARAMS = dict(
    colsample_bytree=0.8,
    learning_rate=0.01,
    max_depth=7,
    n_estimators=500,
    subsample=1.0,
)
BEST_THRESHOLD = 0.65

def load_data():
    path = kagglehub.dataset_download("anmolkumar/health-insurance-cross-sell-prediction")
    df = pd.read_csv(f"{path}/train.csv")
    return df

def engineer_features(df):
    df = df.copy()
    df['Annual_Premium_log'] = np.log1p(df['Annual_Premium'])
    df['Premium_per_Vintage'] = np.where(df['Vintage'] != 0, df['Annual_Premium'] / df['Vintage'], 0)
    df['Damaged_and_Uninsured'] = ((df['Vehicle_Damage'] == 'Yes') & (df['Previously_Insured'] == 0)).astype(int)

    return df

def encode_cat(df):
    df = df.copy()
    encoder = OrdinalEncoder(categories=[VEHICLE_AGE_ORDER])
    df['Vehicle_Age'] = encoder.fit_transform(df[['Vehicle_Age']])
    df['Vehicle_Damage'] = (df['Vehicle_Damage'] == 'Yes').astype(int)
    df['Gender'] = (df['Gender'] == 'Female').astype(int)

    return df

def smoothed_target_encode(train_col, train_target, smoothing=20):
    global_rate = train_target.mean()
    stats = train_target.groupby(train_col).agg(['mean', 'count'])
    smoothed = (stats['count'] * stats['mean'] + smoothing * global_rate) / (stats['count'] + smoothing)
    return smoothed, global_rate

def apply_target_encoding(df, region_rates, channel_rates, global_rate):
    df = df.copy()
    df['Region_Response_Rate'] = df['Region_Code'].map(region_rates).fillna(global_rate)
    df['Channel_Response_Rate'] = df['Policy_Sales_Channel'].map(channel_rates).fillna(global_rate)
    df = df.drop(columns=['Annual_Premium', 'Region_Code', 'Policy_Sales_Channel'])
    return df

def train_model(df):
    df = engineer_features(df)
    df = encode_cat(df)
 
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].copy()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=2
    )
 
    region_rates, region_global_rate = smoothed_target_encode(X_train['Region_Code'], y_train, smoothing=20)
    channel_rates, _ = smoothed_target_encode(X_train['Policy_Sales_Channel'], y_train, smoothing=20)
    global_rate = y_train.mean()
 
    X_train = apply_target_encoding(X_train, region_rates, channel_rates, global_rate)
    X_test = apply_target_encoding(X_test, region_rates, channel_rates, global_rate)
 
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
 
    neg, pos = y_train.value_counts()
    weight_ratio = neg / pos
 
    model = XGBClassifier(
        **BEST_PARAMS,
        scale_pos_weight=weight_ratio,
        eval_metric='logloss',
        random_state=2,
    )
    model.fit(X_train_sc, y_train)
 
    y_proba_test = model.predict_proba(X_test_sc)[:, 1]
    y_pred_test = (y_proba_test >= BEST_THRESHOLD).astype(int)
 
    print('--- Test metrics ---')
    print(f'Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred_test):.4f}')
    print(f'Recall:    {recall_score(y_test, y_pred_test):.4f}')
    print(f'F1 Score:  {f1_score(y_test, y_pred_test):.4f}')
    print(f'ROC-AUC:   {roc_auc_score(y_test, y_proba_test):.4f}')
    print(f'PR-AUC:    {average_precision_score(y_test, y_proba_test):.4f}')
 
    artifacts = {
        'model': model,
        'scaler': scaler,
        'region_rates': region_rates,
        'channel_rates': channel_rates,
        'global_rate': global_rate,
        'threshold': BEST_THRESHOLD,
        'num_cols': NUM_COLS,
        'cat_cols': CAT_COLS,
    }
    return artifacts

def save_model(filename, artifacts):
    with open(filename, 'wb') as f_out:
        pickle.dump(artifacts, f_out)
    print(f'model saved to {filename}')

if __name__ == '__main__':
    df = load_data()
    artifacts = train_model(df)
    save_model(OUTPUT_FILE, artifacts)