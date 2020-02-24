
#%% Linear Ridge

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='ticks')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# parameters
cv_splits = 2
test_size = 0.2

#hyperparameters
hyper_parameters = {
  #'svc__kernel': ['linear', 'rbf'],
  #'svc__C': [1, 10]
  #'ridge__alpha':np.linspace(-4, 0)
}

# 1. load data
data = pd.read_csv("data/tom2019.preprocessed.csv")

# y
response = data[['correct']].copy()
rt = data[['rt']].values.reshape(-1,1)

X_extended = data[['n_recent_targets','n_recent_lures','n_recent_repetitions','N','n_targets','n_lures', 'n_targets','n_lures','N']].copy()
X_basic = data[['n_targets','N']].copy()


X_train, X_test, y_train, y_test = train_test_split(X_extended, response, test_size=0.2, stratify=None)

# analysis steps
steps = [
  ('impute', SimpleImputer(strategy='constant')),
  #('onehot', OneHotEncoder(sparse=False)),
  ('scale', StandardScaler()),
  #('ridge', Ridge())
  ('rfc', RandomForestClassifier())
  #('svc', SVC())
]

pipeline = Pipeline(steps)

grid = GridSearchCV(pipeline, hyper_parameters, cv=cv_splits, scoring='roc_auc', n_jobs=-1, verbose=2)

grid.fit(X_train, y_train)


#y_pred = grid.predict(X_test)
#error = np.sqrt(mean_squared_error(y_test, y_pred))


score = grid.score(X_test, y_test)

print(f"{grid.scoring} = {score}")

#plot_precision_recall_curve(grid, X_test, y_test)
plot_roc_curve(grid, X_test, y_test)
plot_confusion_matrix(grid, X_test, y_test)
