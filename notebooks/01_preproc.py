#%% [markdown]
# # ICOM Preprocessing
# This notebook demonstrate how to load and preprocess the two datasets provided in ICOM project.

#%% preprocess

%reset

from IPython.display import display

from icom.preprocessing import CL2016Preprocessor, Tom2019Preprocessor

cl2016_df = CL2016Preprocessor('data/cl2016_nb.csv').dataframe()
tom2019_df = Tom2019Preprocessor('data/tom2019.csv').dataframe()

display(cl2016_df.correct.describe())
display(tom2019_df.correct.describe())
display(tom2019_df['n_targets'].describe())
display(tom2019_df.n_recent_lures.describe())

#%% exploratory graphs

import seaborn as sns
sns.set(style='ticks')

#sns.pairplot(cl2016_df, vars=['n_targets','n_lures','n_corrects','rt'], hue='N')

# how rt and accuracy are affected by n_recent_targets, n_recent_lures, n_recent_repetitions
sns.pairplot(cl2016_df, x_vars=['n_recent_targets', 'n_recent_lures', 'n_recent_repetitions'], y_vars= ['n_corrects','rt', 'n_recent_corrects'], hue='N', kind='reg')

sns.pairplot(tom2019_df, x_vars=['n_recent_targets', 'n_recent_lures', 'n_recent_repetitions'], y_vars= ['n_corrects','rt', 'n_recent_corrects'], hue='N', kind='reg')