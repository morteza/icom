#%%

%reset

from IPython.display import display

from icom.preprocessing import CL2016Preprocessor

cl2016_df = CL2016Preprocessor('data/cl2016_nb.csv').dataframe()

# %%
