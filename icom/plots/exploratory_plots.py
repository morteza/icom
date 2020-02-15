#%%
from ggplot import *


ggplot(aes(x='date', y='beef'), data=meat) +\
    geom_line() +\
    stat_smooth(colour='blue', span=0.2)  


"""   ggplot() + 
  geom_bar(aes(trial.image,n),stat='identity') + 
  facet_grid(block.pattern_type ~ block.nback) + 
  labs(x='Letter presented', y='Relative frequency') + 
  theme(axis.ticks.y = element_blank(),
        axis.text.y = element_blank())
 """
