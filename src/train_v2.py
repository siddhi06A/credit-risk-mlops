import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

mlflow.set_tracking_uri('http://host.docker.internal:5001')
mlflow.set_experiment('credit_risk_model_v2')

print('Generating improved training data...')

# Create CLEARLY DISTINCT risk groups
np.random.seed(42)
n_samples = 5000

# LOW RISK customers (70%) - good credit profile
low_risk_count = 3500
low_income = np.random.normal(75000, 15000, low_risk_count)
low_debt = np.random.normal(5000, 2000, low_risk_count)
low_age = np.random.normal(42, 10, low_risk_count)
low_credit = np.random.normal(780, 40, low_risk_count)
low_employed = np.random.normal(10, 3, low_risk_count)

# HIGH RISK customers (30%) - poor credit profile
high_risk_count = 1500
high_income = np.random.normal(32000, 10000, high_risk_count)
high_debt = np.random.normal(22000, 7000, high_risk_count)
high_age = np.random.normal(23, 4, high_risk_count)
high_credit = np.random.normal(520, 50, high_risk_count)
high_employed = np.random.normal(2, 1.5, high_risk_count)

# Combine
income = np.concatenate([low_income, high_income])
debt = np.concatenate([low_debt, high_debt])
age = np.concatenate([low_age, high_age])
credit_history = np.concatenate([low_credit, high_credit])
employed_years = np.concatenate([low_employed, high_employed])

# Create target (1 = default, 0 = no default)
default = np.concatenate([np.zeros(low_risk_count), np.ones(high_risk_count)])

# Add some noise to make it realistic (10% of low risk default, 10% of high risk don't default)
noise_idx = np.random.choice(len(default), int(len(default)*0.1), replace=False)
default[noise_idx] = 1 - default[noise_idx]

data = pd.DataFrame({
    'income': income,
    'debt': debt,
    'age': age,
    'credit_history': credit_history,
    'employed_years': employed_years,
    'default': default
})

feature_cols = ['income', 'debt', 'age', 'credit_history', 'employed_years']
X = data[feature_cols]
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training Random Forest model...')
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_metrics({
        'accuracy': accuracy,
        'auc': auc
    })
    mlflow.log_params({
        'n_estimators': 150,
        'max_depth': 12
    })
    
    for col, imp in zip(feature_cols, model.feature_importances_):
        mlflow.log_metric(f'importance_{col}', imp)
    
    mlflow.sklearn.log_model(model, 'credit_risk_model')
    
    print(f'\n✅ Model trained successfully!')
    print(f'   Accuracy: {accuracy:.4f}')
    print(f'   AUC: {auc:.4f}')
    print(f'\nFeature Importance:')
    for col, imp in zip(feature_cols, model.feature_importances_):
        print(f'   {col}: {imp:.4f}')
    
    # Test the high-risk customer
    test_high_risk = [[25000, 20000, 22, 500, 1]]
    test_df = pd.DataFrame(test_high_risk, columns=feature_cols)
    proba = model.predict_proba(test_df)[0]
    print(f'\n🔍 Test: High Risk Customer (25000 income, 20000 debt, age 22, credit 500)')
    print(f'   Risk Score: {proba[1]:.4f} ({proba[1]*100:.1f}%)')
    risk_level = "🔴 HIGH" if proba[1] > 0.6 else "🟡 MEDIUM" if proba[1] > 0.3 else "🟢 LOW"
    print(f'   Risk Level: {risk_level}')
    
    # Test the low-risk customer
    test_low_risk = [[85000, 5000, 40, 800, 12]]
    test_df2 = pd.DataFrame(test_low_risk, columns=feature_cols)
    proba2 = model.predict_proba(test_df2)[0]
    print(f'\n🔍 Test: Low Risk Customer (85000 income, 5000 debt, age 40, credit 800)')
    print(f'   Risk Score: {proba2[1]:.4f} ({proba2[1]*100:.1f}%)')
    risk_level2 = "🔴 HIGH" if proba2[1] > 0.6 else "🟡 MEDIUM" if proba2[1] > 0.3 else "🟢 LOW"
    print(f'   Risk Level: {risk_level2}')