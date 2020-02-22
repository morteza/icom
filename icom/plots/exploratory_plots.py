#%%
from IPython.display import display
import matplotlib.pyplot as plt
from ggplot import *

p = ggplot(aes(x='date', y='beef'), data=meat) +\
  geom_line() +\
  stat_smooth(colour='blue', span=0.2)

ggplot(mpg, aes(x='cty')) + \
    geom_histogram() + \
    xlab("City MPG (Miles per Gallon)") + \
    ylab("# of Obs")

display(p)