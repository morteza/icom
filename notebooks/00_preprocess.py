#%% [markdown]
# # ICOM Preprocessing
# This notebook demonstrate how to load and preprocess the two datasets provided in ICOM project.

#%% preprocess

# reset Jupyter variables. There seems to be a bug that prevents python to recompile imported class.
%reset -f

import pandas as pd
from IPython.display import display
from icom.preprocessing import CL2016Preprocessor, Tom2019Preprocessor

cl2016_df = CL2016Preprocessor('data/cl2016_nb.csv').dataframe()
tom2019_df = Tom2019Preprocessor('data/tom2019.csv').dataframe()

display(cl2016_df.correct.describe())
display(tom2019_df.correct.describe())
display(tom2019_df['n_targets'].describe())
display(tom2019_df.n_recent_lures.describe())

cl2016_df.to_csv('data/cl2016_nb.preprocessed.csv')
tom2019_df.to_csv('data/tom2019.preprocessed.csv')

#%% some basic exploratory reports

# within subject check: if each subject played only a single condition (N)
(tom2019_df.groupby('participant').apply(lambda s: len(s.N.value_counts())) == 1).any()


#%% exploratory graphs

import seaborn as sns
sns.set(style='ticks')

def plot_outcomes_against_params(df: pd.DataFrame):
  "how rt and accuracy are affected by n_recent_targets, n_recent_lures, n_recent_repetitions"

  sns.pairplot(df, x_vars=['n_recent_targets', 'n_recent_lures', 'n_recent_repetitions'], y_vars= ['n_corrects','rt', 'n_recent_corrects'], hue='N', kind='reg')

plot_outcomes_against_params(cl2016_df)
plot_outcomes_against_params(tom2019_df)