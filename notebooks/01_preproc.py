#%% [markdown]
# # ICOM Preprocessing
# This notebook demonstrate how to load and preprocess the two datasets provided by ICOM project.

#%%

%reset

from IPython.display import display

from icom.preprocessing import CL2016Preprocessor, Tom2019Preprocessor

cl2016_df = CL2016Preprocessor('data/cl2016_nb.csv').dataframe()
tom2019_df = Tom2019Preprocessor('data/tom2019.csv').dataframe()

display(tom2019_df.correct.describe())
display(tom2019_df['n_targets'].describe())
display(tom2019_df.n_recent_lures.describe())
