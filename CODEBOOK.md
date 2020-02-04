
# Variables

The following variables are added to the original dataset.

| Variable |  Description |
|---|---|
| `x_t`   | number of targets in the block |
| `x_l`   | number of lures in the block |
| `x_tl`  | number  of targets in previous 8 trials, including current trial |
| `x_ll`  | number of lures in the recent 8 trials, including current trial |
| `x_s`   | skewness of the stimuli frequency distribution of the block |
| `x_sl`  | skewness of the stimuli frequency distribution, limited to previous 8 trials. |
| `x_v`   | Block-level vocabulary size, i.e. number of unique stimuli in the block |
| `x_vl`  | Number of unique stimuli in previous 8 trials, including current trial. |
| `x_repetition` | ... |
| `x_recent_repetition` | ... |
| `x_lumpiness`  | ... |
| `x_recent_lumpiness`  | ... |
