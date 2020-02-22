# 

import pandas as pd
import numpy as np

class Tom2019Preprocessor():
  """Loads and preprocesses Tom2019 N-back dataset.

  Parameters
  ----------
  csv_path : string, default=None
  window_size: int, default=4

  Examples
  --------
  tom2019_df = Tom2019Preprocessor().as_dataframe()

  """

  def __init__(self, csv_path:str=None, window_size:int=4):
    super().__init__()
    self.window_size = window_size
    self.data = self.__preprocess(csv_path)

  def dataframe(self):
    return self.data

  def __preprocess(self, csv_path):
    
    data = pd.read_csv(csv_path, index_col=0)

    data.rename(columns={
      "block.nback": "N",
      "block.number": "block",
      "meta.uniqueid": "participant",
      "trial.image": "stimulus",
      "response.time": "rt",
      "response.key": "choice",
      "trial.target_response": "stimulus_type",
      "block.label": "condition"
    }, inplace=True)

    data.drop(columns=['letter_1back','letter_2back','after_target_lure','before_target_lure'],inplace=True)

    data['condition'] = data.condition.map({
      '1back_test': '1-back',
      '2back_test': '2-back',
      '3back_test': '3-back'
    })

    data['choice'] = data.choice.map({
      's': 'target',
      'd': 'non-target'
    })

    data['stimulus_type'] = data.stimulus_type.map({
      's': 'target',
      'd': 'non-target'
    })

    burn_ins = data['trial.index'] <= data.N
    data.loc[burn_ins,'stimulus_type'] = 'burn-in'

    data['correct'] = (data.stimulus_type.str.lower() == data.choice.str.lower()) & (data.stimulus_type.str.lower() != 'burn-in')


    categorical_cols = ['participant','block','stimulus','stimulus_type','choice']
    data[categorical_cols] = data[categorical_cols].astype('category')

    grouped  = data.groupby(['participant','block','N'])

    # number of targets per block
    data['n_targets'] = grouped.stimulus_type.transform(lambda s: s.value_counts()['target']
    )

    # number of correct responses per block
    #FIXME: what happens when all responses are incorrect
    data['n_corrects'] = grouped.correct.transform(lambda s: s.value_counts()[True])

    data['lure'] = grouped.apply(self.__find_lures)

    data['n_lures'] =  grouped.lure.transform(
      lambda s: s.count() - s.value_counts()[False]
    )

    data['n_recent_repetitions'] = grouped.apply(self.__count_recent_repetitions)
    data['n_recent_targets'] = grouped.apply(self.__count_recent_targets)
    data['n_recent_lures'] = grouped.apply(self.__count_recent_lures)
    data['n_recent_corrects'] = grouped.apply(self.__count_recent_corrects)

    return data

  def __find_lures(self,df):
    """Extract lures from a dataset and return a same-size dataframe with lure trials marked."""

    N = df.N.values[0]
    L1 = df.stimulus.shift(N+1)
    L2 = df.stimulus.shift(N-1)
    non_targets = (df.stimulus_type != 'target')
    df['lures'] = (L1.eq(df.stimulus) | L2.eq(df.stimulus)) & non_targets
    return df[['lures']]

  def __count_recent_repetitions(self,df):
    df['n_recent_repetitions'] = df.stimulus.cat.codes.rolling(self.window_size).apply(
      lambda stim: len(stim) - stim.nunique(),
      raw=False
    )
    return df[['n_recent_repetitions']]

  def __count_recent_targets(self,df):

    target_cat_code = df.stimulus_type.cat.categories.get_loc('target')

    df['n_recent_targets'] = df.stimulus_type.cat.codes.rolling(self.window_size).apply(
      lambda stim: (stim == target_cat_code).sum(),
      raw=False
    )
    return df[['n_recent_targets']]

  def __count_recent_lures(self,df):
    df['n_recent_lures'] = df.lure.rolling(self.window_size).apply(
      lambda l: (l).sum(),
      raw=False
    )
    return df[['n_recent_lures']]

  def __count_recent_corrects(self,df):
    df['n_recent_corrects'] = df.correct.rolling(self.window_size).apply(
      lambda l: (l).sum(),
      raw=False
    )
    return df[['n_recent_corrects']]
