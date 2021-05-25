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


def splitdata_test(data):
    sentence=[]
    for i in data:
        sentence.append(i['sentence'])

    return sentence


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
    '--model_dir', type=str, required=True, help='Path to model')

    parser.add_argument(
    '--save_name', type=str, required=True, help='Path to the result file that you will be saving')

    FLAGS = parser.parse_args()

    
    
    tokenizer = AutoTokenizer.from_pretrained(os.getcwd()+"/models/roberta-base/tokenizer")
    
    trainingdata_en=read("traindata_en_sampled.json")
    testingdata_en=read("test_en.json")
    
    train_texts,train_labels=splitdata(trainingdata_en)
    val_texts=splitdata_test(testingdata_en)

    train_encodings = tokenizer(train_texts, truncation=True, max_length=512,padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=512,padding=True,return_tensors="pt")
    
    
    train_dataset = task1_2_Dataset(train_encodings, train_labels)
    val_dataset = task1_2_Dataset(val_encodings, np.zeros((len(val_texts)),dtype=int))
    
    
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
        
        
    
    training_args = TrainingArguments(
        output_dir='./roberta-base_FINETUNE_BASE',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./log_roberta-base_FINETUNE_BASE',            # directory for storing logs
        logging_steps=10,    
        evaluation_strategy="epoch",
        save_strategy="epoch"
    
    )
    model = AutoModelForSequenceClassification.from_pretrained(os.getcwd()+FLAGS.model_dir)
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )
    
    
    
    arr=trainer.predict(val_dataset)
    
    import pickle
    with open(FLAGS.save_name, 'wb') as handle:
        pickle.dump(arr, handle)
    
    
    

