import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

import pickle
import os

# Seed for random functions
SEED = 1

df_features = pd.read_csv('data/training_set_features.csv')
df_targets = pd.read_csv('data/training_set_labels.csv')

################## DATA PREPARATION ##################

# Transform certain string columns to remove special characters
education_map = {'< 12 Years': 'Lt 12 Years'}
age_group_map = {'55 - 64 Years': '55 to 64', '35 - 44 Years': '35 to 44', '18 - 34 Years': '18 to 34', '65+ Years': '65 plus', '45 - 54 Years': '45 to 54'}
income_poverty_map = {'<= $75,000, Above Poverty': 'Above Poverty Lte 75k', '> $75,000': 'Above Poverty Gt 75k'}

df_features['education'] = df_features['education'].map(education_map).fillna(df_features['education'])
df_features['age_group'] = df_features['age_group'].map(age_group_map).fillna(df_features['age_group'])
df_features['income_poverty'] = df_features['income_poverty'].map(income_poverty_map).fillna(df_features['income_poverty'])

# Standardize values for string columns
string_columns = ['age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'census_msa']
for c in string_columns:
    df_features[c] = df_features[c].str.lower().str.replace(',', '').str.replace(' ', '_')

# Get all float-valued columns
float_columns = list(df_features.dtypes[df_features.dtypes == 'float64'].index)

# Transform all float-valued columns to type string (except nan)
for c in float_columns:
    df_features[c] = df_features[c].astype(str)[df_features[c].notnull()]

# Drop the columns with close to 50% data missing
df_features = df_features.drop(['health_insurance', 'employment_industry', 'employment_occupation'], axis=1)

# Split the data into training and testing subsets (80-20 split)
features_train, features_test, targets_train, targets_test = train_test_split(df_features, df_targets, test_size=0.2, random_state=SEED)

# Reset indices after shuffling
features_train = features_train.reset_index(drop=True).drop('respondent_id', axis=1)
features_test = features_test.reset_index(drop=True).drop('respondent_id', axis=1)
targets_train = targets_train.reset_index(drop=True).drop('respondent_id', axis=1)
targets_test = targets_test.reset_index(drop=True).drop('respondent_id', axis=1)

# Create SimpleImputer object for mode imputation
imputer = SimpleImputer(strategy='most_frequent')

# Impute all missing values in both training and testing feature dataframes
features_train = pd.DataFrame(imputer.fit_transform(features_train), columns=features_train.columns)
features_test = pd.DataFrame(imputer.transform(features_test), columns=features_test.columns)

################## TRAIN H1N1 MODEL ##################

# Initialize DictVectorizer for one-hot encoding categorical features
dv_h1n1 = DictVectorizer(sparse=False)

# Convert feature dataframes (with subset of most important features) to a list of dictionaries, where each dictionary represents a row
dicts_train_h1n1 = features_train.to_dict(orient='records')
dicts_test_h1n1 = features_test.to_dict(orient='records')

# Fit the DictVectorizer on the training set features, and transform the training and test set features
X_train_h1n1 = dv_h1n1.fit_transform(dicts_train_h1n1)
X_test_h1n1 = dv_h1n1.transform(dicts_test_h1n1)

# Get NumPy arrays corresponding to training and test set targets
y_train_h1n1 = targets_train['h1n1_vaccine'].values
y_test_h1n1 = targets_test['h1n1_vaccine'].values

# Initialize XGBClassifier model
xgbc_h1n1 = XGBClassifier(random_state=SEED)

# Create the KFold object with n=5 splits 
kf_h1n1 = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Parameter grid to search over
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10, 15],
    'eta': [0.05, 0.1],
    'grow_policy': ['depthwise', 'lossguide']
}

# Create the RandomizedSearchCV object
print('Training H1N1 Model\n')
xgbc_cv_h1n1 = RandomizedSearchCV(xgbc_h1n1, params, cv=kf_h1n1, scoring='roc_auc', verbose=2, random_state=SEED)
print('\n')

# Perform randomized search over the parameter space
xgbc_cv_h1n1.fit(X_train_h1n1, y_train_h1n1)

# Print the best parameters and AUC score we found
print("Tuned XGBoost Classifier Parameters: {}".format(xgbc_cv_h1n1.best_params_))
print("Tuned XGBoost Classifier Best Accuracy Score: {}".format(xgbc_cv_h1n1.best_score_))
print("\n")

################## TRAIN SEASONAL FLU MODEL ##################

# Initialize DictVectorizer for one-hot encoding categorical features
dv_seasonal = DictVectorizer(sparse=False)

# Convert feature dataframes (with subset of most important features) to a list of dictionaries, where each dictionary represents a row
dicts_train_seasonal = features_train.to_dict(orient='records')
dicts_test_seasonal = features_test.to_dict(orient='records')

# Fit the DictVectorizer on the training set features, and transform the training and test set features
X_train_seasonal = dv_seasonal.fit_transform(dicts_train_seasonal)
X_test_seasonal = dv_seasonal.transform(dicts_test_seasonal)

# Get NumPy arrays corresponding to training and test set targets
y_train_seasonal = targets_train['seasonal_vaccine'].values
y_test_seasonal = targets_test['seasonal_vaccine'].values

# Initialize XGBClassifier model
xgbc_seasonal = XGBClassifier(random_state=SEED)

# Create the KFold object with n=5 splits 
kf_seasonal = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Parameter grid to search over
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10, 15],
    'eta': [0.05, 0.1],
    'grow_policy': ['depthwise', 'lossguide']
}

# Create the RandomizedSearchCV object
print('Training Seasonal Flu Model\n')
xgbc_cv_seasonal = RandomizedSearchCV(xgbc_seasonal, params, cv=kf_seasonal, scoring='roc_auc', verbose=2, random_state=SEED)
print('\n')

# Perform randomized search over the parameter space
xgbc_cv_seasonal.fit(X_train_seasonal, y_train_seasonal)

# Print the best parameters and AUC score we found
print("Tuned XGBoost Classifier Parameters: {}".format(xgbc_cv_seasonal.best_params_))
print("Tuned XGBoost Classifier Best Accuracy Score: {}".format(xgbc_cv_seasonal.best_score_))
print("\n")

################## EVALUATE MODELS ON TEST DATA ##################

# Get the predicated probability scores on the test set
y_pred_h1n1 = xgbc_cv_h1n1.predict_proba(X_test_h1n1)[:, 1]
y_pred_seasonal = xgbc_cv_seasonal.predict_proba(X_test_seasonal)[:, 1]

# Place predicted probabilities for the h1n1 and seasonal flu targets into a DataFrame
pred_targets_test = pd.DataFrame(np.array([y_pred_h1n1, y_pred_seasonal]).T, columns=['h1n1_vaccine', 'seasonal_vaccine'])

# Compute the macro (unweighted average) AUC score
print(f"Mean ROC AUC on test data: {roc_auc_score(targets_test, pred_targets_test, average='macro')}")
print("\n")

################## SAVE THE FINAL MODELS AND DICTVECTORIZERS ##################

output_file = f'bin/final_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv_h1n1, xgbc_cv_h1n1, dv_seasonal, xgbc_cv_seasonal), f_out)

print(f'The final model is saved to {os.path.abspath(output_file)}')