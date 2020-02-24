
#%% Linear Ridge

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge

#y = tom2019_df[['correct','rt']].copy()
y = tom2019_df[['rt','correct']].values.reshape(-1,2)
X1 = tom2019_df[['n_recent_targets','n_recent_lures','n_recent_repetitions','N','n_targets','n_lures']].copy()
X2 = tom2019_df[['n_targets','N']].copy()

steps = [
  ('impute', SimpleImputer(strategy='constant')),
  #('onehot', OneHotEncoder(sparse=False)),
  ('scale', StandardScaler()),
  ('ridge', Ridge())
]

pipeline = Pipeline(steps)

#hyperparameters
parameters = {'ridge__alpha':np.linspace(-4, 0)}

cv = GridSearchCV(pipeline, parameters, cv=5)

cv.fit(X1, y)
y_pred1 = cv.predict(X1)

cv.fit(X2, y)
y_pred2 = cv.predict(X2)

error1 = np.sqrt(mean_squared_error(y, y_pred1))
error2 = np.sqrt(mean_squared_error(y, y_pred2))

print(f"RMSE1={error1}, RMSE2={error2}")


#%%

from IPython.display import display

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_predict

# data



X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]])
y = [True,False,True,True,True,True,True,False]

# hyper parameters
max_comp = X.shape[1]
cv_splits = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


def find_optimal_comp(X, y, max_comp):
  """Finds number of compoenents that produces minimum MSE error.
  """

  errors = []
  for c in np.arange(1,max_comp):
    pls = PLSRegression(n_components=c)

    y_cv = cross_val_predict(pls, X, y, cv=cv_splits)
    errors.append(mean_squared_error(y, y_cv))

  return np.argmin(errors) + 1


# find optimal number of PLS components
optimal_comp = find_optimal_comp(X_train, y_train, max_comp)

# define and fit PLS model
pls = PLSRegression(n_components=optimal_comp)
pls.fit(X_train, y_train)

#TODO separate X_train from X_test
y_predicted = cross_val_predict(pls, X, y, cv=cv_splits)

# calculate errors
auc_error = roc_auc_score(y, y_predicted)
r2_error = r2_score(y, y_predicted)
mse_error = mean_squared_error(y, y_predicted)
fpr, tpr, _ = roc_curve(y, y_predicted)
roc_auc = auc(fpr, tpr)

display('AUC = %.3f' % auc_error)
display('R2 = %.3f' % r2_error)
display('MSE = %.3f' % mse_error)


#%%
# plot ROC
plt.figure()
lw = 2
plt.plot(fpr*100, tpr*100, label=f'Extended Model (AUC = {roc_auc*100:.2f}%)')

# draw auc=50% line
plt.plot([0, 100], [0, 100], linestyle='--')

plt.xlim([0.0, 100])
plt.ylim([0.0, 105])

plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')

plt.legend()
display(plt)
