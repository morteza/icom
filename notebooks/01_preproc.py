#%% [markdown]
# # ICOM Preprocessing
# This notebook demonstrate how to load and preprocess the two datasets provided in ICOM project.

#%%

%reset

from IPython.display import display

from icom.preprocessing import CL2016Preprocessor, Tom2019Preprocessor

cl2016_df = CL2016Preprocessor('data/cl2016_nb.csv').dataframe()
#tom2019_df = Tom2019Preprocessor('data/tom2019.csv').dataframe()

display(cl2016_df['correct'])
display(f"CL corrects: {cl2016_df['correct'].sum()}")

#display(tom2019_df.correct.describe())
#display(tom2019_df['n_targets'].describe())
#display(tom2019_df.n_recent_lures.describe())

#%%
import seaborn as sns
#sns.pairplot(cl2016_df)

display(cl2016_df)