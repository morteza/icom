
# Additional Variables

Along with the original parameters of each trial, the following parameters are extracted and added to the original N-back datasets.

| Variable |  Description |
|---|---|
| `N`     | Number of trials to look back |
| `x_t`   | number of targets in the block |
| `x_l`   | number of lures in the block. Pre-lures and post-lures are both considered. |
| `x_s`   | skewness of the stimuli frequency distribution of the block. It's the maximum deviation from uniform distribution. |
| `x_u`   | Block-level lumpiness is the maximum number of repetition in the whole block. |
| `x_v`   | Block-level vocabulary size, i.e. number of unique stimuli in the block |
| `x_s8`  | skewness of the stimuli frequency distribution, limited to previous 8 trials. |
| `x_t8`  | number  of targets in previous 8 trials, including current trial |
| `x_l8`  | number of lures in the recent 8 trials, including current trial |
| `x_v8`  | Number of unique stimuli in previous 8 trials, including current trial. |
| `x_u8`  | Lumpiness is the maximum number of repetition in recent trials. |

Note that some parameters contain a digit at the end of their name. These columns contain a local variable which is calculated by combining a number of recent trials. For example `8` in `x_t8` shows that the parameter, i.e., number of targets, was calculated using 8 previous trials and shows the number of targets in the previous 8 trials.

- [ ] TODO: explain that local variables do not contain current trial.
