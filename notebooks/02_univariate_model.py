
#%% Linear Ridge

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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


def fit_classifier(X, y, cv_splits = 3, test_ratio = 0.2):
  """Train a univariate model for X and y, with test/train split and cross-validationand returns an estimator."""
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=None)

  # pipeline steps
  steps = [
    ('impute', SimpleImputer(strategy='constant')),
    #('onehot', OneHotEncoder(sparse=False)),
    ('scale', StandardScaler()),
    #('ridge', Ridge())
    ('rfc', RandomForestClassifier())
    #('svc', SVC())
  ]
  
  #hyperparameters
  hyper_parameters = {
    #'svc__kernel': ['linear', 'rbf'],
    #'svc__C': [1, 10]
    #'ridge__alpha':np.linspace(-4, 0)
  }

  pipeline = Pipeline(steps)

  # grid search for hyper parameters with cross-validation
  grid = GridSearchCV(pipeline, hyper_parameters, cv=cv_splits, scoring='roc_auc', n_jobs=-1, verbose=1)

  grid.fit(X_train, y_train)
  score = grid.score(X_test, y_test)

  #y_pred = grid.predict(X_test)
  #error = np.sqrt(mean_squared_error(y_test, y_pred))

  print(f"{grid.scoring} = {score}")

  #plot_precision_recall_curve(grid, X_test, y_test)
  #plot_confusion_matrix(grid, X_test, y_test)

  return grid, X_train, X_test, y_train, y_test


#%% Actual analysis
# ---------------------------------------------------
# 1. load data
#data = pd.read_csv("data/cl2016_nb.preprocessed.csv")
data = pd.read_csv("data/tom2019.preprocessed.csv")

# 2. output
y_response = data[['correct']].copy()
y_rt = data[['rt']].values.reshape(-1,1)

# 3. input
X_extended = data[['n_recent_targets','n_recent_lures','n_recent_repetitions','N','n_targets','n_lures', 'n_targets','n_lures','N']].copy()
X_basic = data[['n_targets','N']].copy()

# 4. fit models
model_extended = fit_classifier(X_extended, y_response)
model_basic = fit_classifier(X_basic, y_response)

# 5. plot ROC
fig, ax = plt.subplots()
plot_roc_curve(model_basic[0], model_basic[2], model_basic[4], ax=ax, name='Basic Model')
plot_roc_curve(model_extended[0], model_extended[2], model_extended[4], ax=ax, name='Extended Model')
plt.show()

