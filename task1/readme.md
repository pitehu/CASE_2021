# Submission for task 1

# List of Files
* train_wsecondpretrain.py: training script
* evaluate.py: evaluation script.
* traindata_en_sampled.json and testdata_en_sampled.json: our train/test split from subtask 2 data
* Second_pretrain.py: script for carrying out second pretraining
* validate_dataset_for_lm.csv: subtask 2 data converted to csv for second pretrain validation

# Packages

We require the following package:

* torch
* transformers
* numpy
* sklearn

# Pretrained Weights
* Link: https://polybox.ethz.ch/index.php/s/1obl9maAHmUgq1t

* Checkpoint_pretrain: the best weight (determined by downstream task) after the second pretrain.

* Checkpoint_submit: the weight of the actual model for our submission.

# Second pretraining
**Warning: this might take very long**
* POLUSA dataset: https://zenodo.org/record/3813664#.XvDMbWhKiUk
