import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import json
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from datetime import datetime

def load_data(filepath):
    with open(filepath) as fp:
        data = json.load(fp)
    
    flat_data = [item for sublist in data for item in sublist] # for fethching the data from the list

    return pd.DataFrame(flat_data)

# Loading Flag Data
train_flag_data = pd.read_csv('senior_ds_test/data/train/train_flag.csv')
test_flag_data = pd.read_csv('senior_ds_test/data/test/test_flag.csv')
# Loading Account Data
account_train_data = load_data('senior_ds_test/data/train/accounts_data_train.json')
account_test_data = load_data('senior_ds_test/data/test/accounts_data_test.json')
# Loading Enquiry Data
enquiry_train_data = load_data('senior_ds_test/data/train/enquiry_data_train.json')
enquiry_test_data = load_data('senior_ds_test/data/test/enquiry_data_test.json')

# Printing all the data just for the clarification

# print(train_flag_data.head())
# print(test_flag_data.head())

# print(account_train_data.head())
# print(account_test_data.head())

# print(enquiry_train_data.head())
# print(enquiry_test_data.head())

# Merging the data

train_merged_data = pd.merge(train_flag_data, account_train_data, on='uid', how='left')
train_data = pd.merge(train_merged_data,enquiry_train_data, on='uid', how='left')
# print(train_data.head())

test_merged_data = pd.merge(test_flag_data, account_test_data, on='uid', how='left')
test_data = pd.merge(test_merged_data,enquiry_test_data, on='uid', how='left')
# print(test_data.head())

# Checking for missing data
missing_columns_train = train_data.columns[train_data.isnull().any()]
# print(missing_columns_train)
missing_columns_test = test_data.columns[test_data.isnull().any()]
# print(missing_columns_test)

# Filling the missing values
train_data.fillna(0,inplace=True)
test_data.fillna(0, inplace=True)

# Plot distribution of numerical features
numerical_features = ['loan_amount', 'amount_overdue', 'enquiry_amt'] 
for feature in numerical_features:
    plt.figure(figsize=(14,6))
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    # plt.show()

# Plot correlation matrix
corr_matrix = train_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()

def feature_engineering(df):
    df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce')
    df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
    df['loan_duration'] = (df['closed_date'] - df['open_date']).dt.days
    df['loan_duration'].fillna((datetime.now() - df['open_date']).dt.days, inplace=True)
    df['enquiry_date'] = pd.to_datetime(df['enquiry_date'], errors='coerce')
    df['days_since_last_enquiry'] = (datetime.now() - df['enquiry_date']).dt.days
    df['days_since_last_enquiry'].fillna(df['days_since_last_enquiry'].max(), inplace=True)
    return df

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)
train_data['payment_hist_string'].astype(str)
test_data['payment_hist_string'].astype(str)

# Do the label encodings for the features
le = LabelEncoder()
train_data['NAME_CONTRACT_TYPE'] = le.fit_transform(train_data['NAME_CONTRACT_TYPE'])
test_data['NAME_CONTRACT_TYPE'] = le.transform(test_data['NAME_CONTRACT_TYPE'])

train_data['credit_type'] = le.fit_transform(train_data['credit_type'].astype(str))
test_data['credit_type'] = le.transform(test_data['credit_type'].astype(str))

train_data['enquiry_type'] = le.fit_transform(train_data['enquiry_type'].astype(str))
test_data['enquiry_type'] = le.transform(test_data['enquiry_type'].astype(str))

# Optimize data types
def optimize_dtypes(df):
    int_columns = df.select_dtypes(include=['int64']).columns
    float_columns = df.select_dtypes(include=['float64']).columns
    df[int_columns] = df[int_columns].astype('int32')
    df[float_columns] = df[float_columns].astype('float32')
    return df

train_data = optimize_dtypes(train_data)
test_data = optimize_dtypes(test_data)
# Process the payment_history_string
# train_data['most_recent_payment'] = train_data['payment_hist_string'].apply(lambda x: (str(x)[-3:]))
# train_data['average_payment_delay'] = train_data['payment_hist_string'].apply(lambda x: np.mean([(str(x)[i:i+3]) for i in range(0, len(x), 3)]))
# test_data['most_recent_payment'] = test_data['payment_hist_string'].apply(lambda x: (str(x)[-3:]))
# test_data['average_payment_delay'] = test_data['payment_hist_string'].apply(lambda x: np.mean([(str(x)[i:i+3]) for i in range(0, len(x), 3)]))

# Target Variable Distribution
plt.title('Distribution of Target Variable before SMOTE')
sns.countplot(x='TARGET', data=train_data)
# plt.show() 
print(train_data['TARGET'].value_counts())
# shows that dataset is very imbalanced 8% for bad loans data only so make the dataset balanced

# Feature Selection by using the importance parameter
X = train_data.drop(columns=['uid', 'TARGET', 'open_date', 'closed_date', 'enquiry_date', 'payment_hist_string'])
y = train_data['TARGET']
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X, y)
# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# plt.figure(figsize=(12, 8))
# sns.barplot(x='importance', y='feature', data=feature_importance_df)
# plt.title('Feature Importance')
# plt.show()

# X = train_data[feature_importance_df['feature'][:10]]  # Select top 20 important features
# y = train_data['TARGET']
# X_test = test_data[feature_importance_df['feature'][:10]]
X_test = test_data.drop(columns=['uid', 'open_date', 'closed_date', 'enquiry_date', 'payment_hist_string'])

# Apply SMOTE and check the distributions
# smote = SMOTE(random_state=42)
# X_balanced, y_balanced = smote.fit_resample(X, y)

# sns.countplot(x=y_balanced)
# plt.title('Distribution of Target Variable After SMOTE')
# plt.show()
# print(y_balanced.value_counts())

# Spliting the data into train and validation and then providing the result on the test data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred)
    print(f'{model_name} Validation ROC AUC Score: {roc_auc}')

# Select the best model based on ROC AUC score
best_model = models['RandomForest']
best_model.fit(X, y)

# Predict on test data
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Save predictions
submission = pd.DataFrame({'uid': test_data['uid'], 'TARGET': test_predictions})
submission.to_csv('final_submission_firstname_lastname.csv', index=False)


# Tried it using Balanced dataset but due to more application memory required could not process it
