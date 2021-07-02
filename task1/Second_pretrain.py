# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:01:27 2021

@author: Superhhu
"""
import os 
import argparse
os.environ['WANDB_DISABLED']="true"


import torch
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments

import json
import numpy as np


def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data

def splitdata(data):
    sentence=[]
    label=[]
    for i in data:
        sentence.append(i['sentence'])
        label.append(i['label'])
    return sentence,label
class task1_2_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

if __name__ == '__main__':
    
    from datasets import load_dataset

    dataset = load_dataset('csv', data_files={'train':['polusa_unbalanced/2017_1.csv','polusa_unbalanced/2017_2.csv',    
                                              'polusa_unbalanced/2018_1.csv','polusa_unbalanced/2018_2.csv',
                                              'polusa_unbalanced/2019_1.csv','polusa_unbalanced/2019_2.csv'
                                             
                                              ]})
    
    valid_dataset = load_dataset('csv', data_files=['validate_dataset_for_lm.csv'])
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained("roberta-base")        
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
                
    
    def tokenize_function(examples):
        return tokenizer(examples["body"])
    
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['id', 'date_publish', 'outlet', 'headline', 'lead', 'body', 'authors', 'domain', 'url', 'political_leaning'])
    
    valid_tokenized_datasets=valid_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['body'])
    
    block_size = 256
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=2000,
        num_proc=4,
    )
    valid_lm_datasets= valid_tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=2000,
        num_proc=4,
    )


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--warmup_steps', type=int, default=10000, help='Name for your run to easier identify it.')
    parser.add_argument(
        '--weight_decay', type=float, default=0.01, help='Place for artifacts and logs')
    parser.add_argument(
        '--learning_rate', type=float, default=5.0e-6, help='Path to dataset')
    parser.add_argument(
        '--output_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument(
        '--logging_dir', type=str, required=True, help='Path to dataset')



    FLAGS = parser.parse_args()


    training_args = TrainingArguments(
        output_dir=FLAGS.output_dir,          # output directory
        num_train_epochs=50,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        adam_epsilon=1e-6,
        adam_beta2=0.98,
        warmup_steps=FLAGS.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=FLAGS.weight_decay,      
        learning_rate=FLAGS.learning_rate,         # strength of weight decay
        logging_dir=FLAGS.logging_dir,            # directory for storing logs
        logging_steps=10,    
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250
    
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=lm_datasets["train"],         # training dataset
        eval_dataset=valid_lm_datasets['train'],
        data_collator=data_collator)           # evaluation dataset    )
    
    trainer.train()
    


