# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:01:27 2021

@author: Superhhu
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import os 
import argparse
os.environ['WANDB_DISABLED']="true"


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--warmup_steps', type=int, default=500, help='Warm up steps.')
    parser.add_argument(
        '--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument(
        '--output_dir', type=str, required=True, help='Path to save model')
    parser.add_argument(
        '--logging_dir', type=str, required=True, help='Path for artifacts and logs')

    parser.add_argument(
        '--training_data', type=str, required=True, help='Path to training dataset')

    parser.add_argument(
        '--testing_data', type=str, required=True, help='Path to testing dataset')

    parser.add_argument(
        '--check_point', type=str, required=True, help='Path to the initialization weight')




    FLAGS = parser.parse_args()




    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    trainingdata_en=read(FLAGS.training_data)
    testingdata_en=read(FLAGS.testing_data)
    
    train_texts,train_labels=splitdata(trainingdata_en)
    val_texts,val_labels=splitdata(testingdata_en)
    
    
    train_encodings = tokenizer(train_texts, max_length=512,truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, max_length=512,truncation=True, padding=True)
    
    
    train_dataset = task1_2_Dataset(train_encodings, train_labels)
    val_dataset = task1_2_Dataset(val_encodings, val_labels)
    
    
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        
        
    model = AutoModelForSequenceClassification.from_pretrained(os.getcwd()+FLAGS.check_point)
    
    training_args = TrainingArguments(
        output_dir=FLAGS.output_dir,          # output directory
        num_train_epochs=25,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=FLAGS.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=FLAGS.weight_decay,      
        learning_rate=FLAGS.learning_rate,         # strength of weight decay
        logging_dir=FLAGS.logging_dir,            # directory for storing logs
        logging_steps=10,    
        evaluation_strategy="epoch",
        save_strategy="epoch"
    
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    


