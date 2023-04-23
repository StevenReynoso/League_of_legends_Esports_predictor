#%%
import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.metrics import confusion_matrix
import xgboost as xgb


data = pd.read_csv("2023_LoL_esports_match_data_from_OraclesElixir.csv", header = 00)

# %%
to_drop = ['url', 'split']
data = data.drop(to_drop, axis=1)
missing_data = data.isnull().sum()
print(missing_data)

# %%
# Split data into features (X) and target (y)
X = data.drop('result', axis=1)
y = data['result']

# Split data into training and testing sets
X_encoded = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#%%
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

#%%
# create an instance of SimpleImputer
imputer = SimpleImputer(strategy='mean')

# fit the imputer to the data
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# %%
print(data.isnull().sum())
print(data.dtypes)
# %%
# Train XGBoost classifier and get feature importances
clf = xgb.XGBClassifier(objective='binary:logistic')
clf.fit(X_train, y_train)
importances = clf.feature_importances_

# Predict using the test data
y_pred = clf.predict(X_test)
#%%
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# %%

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
print("Confusion Matrix:")
print("TN: {}  FP: {}".format(cm[0][0], cm[0][1]))
print("FN: {}  TP: {}".format(cm[1][0], cm[1][1]))
# %%
