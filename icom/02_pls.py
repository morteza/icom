
#%%

from IPython.display import display

import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

# hyper parameters
max_comp = 10
cv_folds = 10
X = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]


def find_optimal_comp(X, y, max_comp):
  """Finds number of compoenents that produces minimum MSE error.
  """

  errors = []
  for c in np.arange(max_comp):
    pls = PLSRegression(n_components=c)

    y_cv = cross_val_predict(pls, X, y, cv=10)
    errors.append(mean_squared_error(y, y_cv))

  return np.argmin(errors) + 1


# find optimal number of PLS components
optimal_comp = find_optimal_comp(X, y, max_comp)

# define and fit PLS model
pls = PLSRegression(n_components=optimal_comp)
pls.fit(X, y)

y_cv = cross_val_predict(pls, X, y, cv=10)

# calculate errors
cv_auc = roc_auc_score(y, y_cv)
cv_r2 = r2_score(y, y_cv)
cv_mse = mean_squared_error(y, y_cv)

display(cv_auc)
display(cv_r2)
display(cv_mse)
