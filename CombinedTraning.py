import Corefunctions as Cf
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import random
import csv
from tqdm import tqdm

import numpy as np

# from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("The Device is:", device)

############################################################################################ PARAMETERS

is_cascade = True
LENGTH = 6
"""
length: length of poem sequences
level:  number of true lines
-----
length = 4, level = 2: [True, Ture, Random, Random]
length = 6, level = 2: [True, Ture, Random, Random, Random, Random]
length = 6, level = 5: [True, Ture, True, Ture, True, Random]
"""

PATH_TO_PRETRAINED_MODEL = ""  # make it empty if you don't have any, empty is like: ""

MAX_LEN = 20 * LENGTH
BATCH_SIZE = 128
EPOCHS = 1

PATH_TO_SONGS = "Data/songs_data.csv"
# PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

print("      *********************************")
print("Your BERT model is: ", PRE_TRAINED_MODEL_NAME)
print("      *********************************")

#############################################################################################

print("Loading the songs file..........")
with open(PATH_TO_SONGS, newline='') as f:
    songs = list(csv.reader(f))

train_songs, val_songs, test_songs = Cf.data_cleaner(songs)

model = Cf.CohClassifier(2, PRE_TRAINED_MODEL_NAME)  # true 1, false 0
if PATH_TO_PRETRAINED_MODEL:
    model.load_state_dict(torch.load(PATH_TO_PRETRAINED_MODEL))
model = model.to(device)

#############################################################################################

loss_fn = nn.CrossEntropyLoss().to(device)

starting_level = 1 if is_cascade else LENGTH - 1
for lvl in range(starting_level, LENGTH):
    if_last = (lvl == LENGTH - 1)  # 1 if it is the last one and 0 if it is not
    batch_size = BATCH_SIZE // (3 * if_last + 1)
    num_epochs = EPOCHS  # the last one has one more epoch to be trained
    print(f" ********* Training length {LENGTH} and level {lvl} and batch size {batch_size} *********")
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    for what_version in range(2 + 2 * if_last):
        version = "same" if what_version % 2 else "all"
        print(f" ********* Training length {LENGTH} and level {lvl} and batch size {batch_size} *********")
        train_data_loader, val_data_loader, test_data_loader = Cf.create_dataloaders(train_songs, val_songs,
                                                                                     test_songs, tokenizer,
                                                                                     batch_size, MAX_LEN,
                                                                                     LENGTH, lvl,
                                                                                     version=version)

        total_steps = len(train_data_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(num_epochs):  # + 1 if it is the last lvl

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            train_acc, train_loss = Cf.train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(train_data_loader) * batch_size
            )

            print(f'\n Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = Cf.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(val_data_loader) * batch_size
            )

            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if if_last:
                torch.save(model.state_dict(),
                           f'Models/Combined/COMBINED Lenghth_{LENGTH} LEVEL_{lvl} epoch_{epoch + 1} model_{val_acc}.bin')
                best_accuracy = val_acc

        test_acc, _ = Cf.eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_data_loader) * batch_size
        )
        print("Test accuracy is: ", test_acc.item())
