#%%
import numpy as np
import pandas as pd

from IPython.display import display

# parameters
window_size = 4

# load data
data = pd.read_csv('data/cl2016_nb.csv',index_col=0)

categorical_cols = ['participant','block','stimulus','stimulus_type','choice']

data[categorical_cols] = data[categorical_cols].astype('category')

#DEBUG
# display(data.info())

data['N'] = np.where(data.condition=='2-back',2,3)
data['correct'] = np.where(
  (data.stimulus_type.str == data.choice.str) & (data.stimulus_type != 'burn-in'),
  True,
  False
  )

grouped  = data.groupby(['participant','block','N'])

# group by participant and block
data['n_targets'] = grouped.stimulus_type.transform(lambda s: s.value_counts()['target']
)

# find lures

def find_lures(x):
  N = x.N.values[0]
  L1 = x.stimulus.shift(N+1)
  L2 = x.stimulus.shift(N-1)
  non_targets = (x.stimulus_type != 'target')
  x['lures'] = (L1.eq(x.stimulus) | L2.eq(x.stimulus)) & non_targets
  return x[['lures']]

data['lure'] = grouped.apply(find_lures)

data['n_lures'] =  grouped.lure.transform(
  lambda s: s.count() - s.value_counts()[False]
)

def count_recent_repetitions(x):
  x['n_recent_repetitions'] = x.stimulus.cat.codes.rolling(window_size).apply(
    lambda stim: len(stim) - stim.nunique(),
    raw=False
  )
  return x[['n_recent_repetitions']]


def count_recent_targets(x):
  target_cat_code = data.stimulus_type.cat.categories.get_loc('target')

  x['n_recent_targets'] = x.stimulus_type.cat.codes.rolling(window_size).apply(
    lambda stim: (stim == target_cat_code).sum(),
    raw=False
  )
  return x[['n_recent_targets']]


def count_recent_lures(x):
  x['n_recent_lures'] = x.lure.rolling(window_size).apply(
    lambda l: (l).sum(),
    raw=False
  )
  return x[['n_recent_lures']]

data['n_recent_repetitions'] = grouped.apply(count_recent_repetitions)
data['n_recent_targets'] = grouped.apply(count_recent_targets)
data['n_recent_lures'] = grouped.apply(count_recent_lures)


# %%
