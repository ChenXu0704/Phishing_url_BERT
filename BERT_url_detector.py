
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import os
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score
import torch
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from keras import models, layers, optimizers
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import create_optimizer
import tensorflow as tf
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
import torch.nn.functional as F
import torch

torch.cuda.empty_cache()



def Preprocessing(path):
    data = pd.read_csv(path, encoding="ISO-8859–1")
    data.dropna()
    data = data.drop_duplicates(subset= ['URL'], keep = 'first')
    X = data['URL']
    y=data['Label']
    return X, y
    

# def Preprocessing():
#     path = "./Dataset.csv"
#     data = pd.read_csv(path, encoding="ISO-8859–1")
#     data.dropna()
#     data = data.drop_duplicates(subset= ['URL'], keep = 'first')
#     X = data['URL']
#     y=[]
#     for label in data['Label']:
#         if label == 'bad':
#             y.append(1)
#         elif label == 'good':
#             y.append(0)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
#     return X_train, X_test, y_train, y_test


# def ObsPreprocessing():
#     path = "./Obs_Dataset.csv"
#     data = pd.read_csv(path, encoding="ISO-8859–1")
#     X = data['URL']
#     y=[]
#     for label in data['Label']:
#         if label == 'bad':
#             y.append(1)
#         elif label == 'good':
#             y.append(0)
#     X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.4)
#     return X_test, y_test


def preprocess_function(examples):
    
    return tokenizer(examples["text"], truncation=True)


X_train, y_train = Preprocessing("./Training.csv")  
X_test, y_test = Preprocessing("./Testing.csv")   
X_st, y_st = Preprocessing("./Shortener_Testing.csv")   
X_rd, y_rd = Preprocessing("./Redirector_Testing.csv")




df_train = pd.DataFrame({'text': X_train, 'label': y_train})
train_dataset = Dataset(pa.Table.from_pandas(df_train))
df_test = pd.DataFrame({'text': X_test, 'label': y_test})
test_dataset = Dataset(pa.Table.from_pandas(df_test))
df_st = pd.DataFrame({'text': X_st, 'label': y_st})
st_dataset = Dataset(pa.Table.from_pandas(df_st))
df_rd = pd.DataFrame({'text': X_rd, 'label': y_rd})
rd_dataset = Dataset(pa.Table.from_pandas(df_rd))


    
URLS = DatasetDict({"train":train_dataset,"test":test_dataset, "st":st_dataset, "rd":rd_dataset}) #   
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_URLS = URLS.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
print("Finish Tokenization!!!!!!!!!!!!")    
    
    
tf_train_set = tokenized_URLS["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = tokenized_URLS["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)   

tf_st_set = tokenized_URLS["st"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_rd_set = tokenized_URLS["rd"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)


batch_size = 32
num_epochs = 5

batches_per_epoch = len(tokenized_URLS["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.compile(optimizer=optimizer) #, loss='binary_crossentropy'
model.fit(tf_train_set, epochs=1)

y_preds = model.predict(tf_validation_set)
y_pred = np.argmax(y_preds.logits, axis=1)
y_st_preds = model.predict(tf_st_set)
y_st_pred = np.argmax(y_st_preds.logits, axis=1) 
y_rd_preds = model.predict(tf_rd_set)
y_rd_pred = np.argmax(y_rd_preds.logits, axis=1)

print("########## Testing Dataset ###########")
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(f1_score(y_test, y_pred, average="macro"))
print(roc_auc_score(y_test, y_pred))

    
print("########## Shortener Obfuscated Dataset ###########")
print(accuracy_score(y_st, y_st_pred))
print(recall_score(y_st, y_st_pred, average="macro"))
print(precision_score(y_st, y_st_pred, average="macro"))
print(f1_score(y_st, y_st_pred, average="macro"))
print(roc_auc_score(y_st, y_st_pred))    



print("########## Redirector Obfuscated Dataset ###########")
print(accuracy_score(y_rd, y_rd_pred))
print(recall_score(y_rd, y_rd_pred, average="macro"))
print(precision_score(y_rd, y_rd_pred, average="macro"))
print(f1_score(y_rd, y_rd_pred, average="macro"))
print(roc_auc_score(y_rd, y_rd_pred))    
