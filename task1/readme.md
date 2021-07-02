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

* Checkpoint_submit: the weight of the actual model for our submission (after finetuning).

# Second pretraining
**Warning: this might take very long**
* acquire the POLUSA dataset: https://zenodo.org/record/3813664#.XvDMbWhKiUk
* run Second_pretrain.py with the appropiate command line argument. Those with default values can be left unchanged. 

# To finetune the model with a saved second pretrained weight:
run train_wsecondpretrain.py with the appropiate command line argument. Those with default values can be left unchanged.

# To finetune the model without second pretraining:
run train_wsecondpretrain.py with the appropiate command line argument. Instead of loading a second-pretrained weight, you
should just load the pretrained model weight. Those with default values can be left unchanged.

# To evaluate the model after finetuning:
run evaluate.py with the appropiate command line argument. 

