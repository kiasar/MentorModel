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

is_sigmoid = True
is_cascade = False
LENGTH = 4
VERSION = "50/50"  # version most be only 'all', 'same', 'start all' and 'start same' or '50/50'.
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
EPOCHS = 100

PATH_TO_SONGS = "Data/songs_data.csv"
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

model = Cf.CohClassifierSigmoid(PRE_TRAINED_MODEL_NAME)  # true 1, false 0
if PATH_TO_PRETRAINED_MODEL:
    model.load_state_dict(torch.load(PATH_TO_PRETRAINED_MODEL))
model = model.to(device)

#############################################################################################

_, val_dataloader_all, test_dataloader_all = Cf.create_dataloaders(train_songs, val_songs,
                                                                   test_songs, tokenizer,
                                                                   BATCH_SIZE, MAX_LEN,
                                                                   LENGTH, LENGTH - 1,
                                                                   version="all")

_, val_dataloader_same, test_dataloader_same = Cf.create_dataloaders(train_songs, val_songs,
                                                                     test_songs, tokenizer,
                                                                     BATCH_SIZE, MAX_LEN,
                                                                     LENGTH, LENGTH - 1,
                                                                     version="same")

if is_sigmoid:
    loss_fn = nn.BCELoss().to(device)
else:
    loss_fn = nn.CrossEntropyLoss().to(device)

test_acc_all, _ = Cf.eval_model(
    model,
    test_dataloader_all,
    loss_fn,
    device,
    len(test_dataloader_all) * BATCH_SIZE,
    is_sigmoid
)
test_acc_same, _ = Cf.eval_model(
    model,
    test_dataloader_same,
    loss_fn,
    device,
    len(test_dataloader_same) * BATCH_SIZE,
    is_sigmoid
)
print(f'Test ALL acc = {test_acc_all}, Test SAME acc = {test_acc_same}')

#############################################################################################

starting_level = 1 if is_cascade else LENGTH - 1
for lvl in range(starting_level, LENGTH):
    is_last_round = (lvl == LENGTH - 1)  # 1 if it is the last one and 0 if it is not
    num_epochs = EPOCHS
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    print(f" ********* Training length {LENGTH}, level {lvl}, version {VERSION}, and batch size {BATCH_SIZE} *********")

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(num_epochs):  # + 1 if it is the last lvl

        train_data_loader, val_data_loader, test_data_loader = Cf.create_dataloaders(train_songs, val_songs,
                                                                                     test_songs, tokenizer,
                                                                                     BATCH_SIZE, MAX_LEN,
                                                                                     LENGTH, lvl,
                                                                                     version=VERSION)

        total_steps = len(train_data_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_acc, train_loss = Cf.train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_data_loader) * BATCH_SIZE,
            is_sigmoid
        )

        print(f'\n Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = Cf.eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(val_data_loader) * BATCH_SIZE,
            is_sigmoid
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')

        if is_last_round or not is_cascade:
            val_acc_all, val_loss_all = Cf.eval_model(
                model,
                val_dataloader_all,
                loss_fn,
                device,
                len(val_dataloader_all) * BATCH_SIZE,
                is_sigmoid
            )
            val_acc_same, val_loss_same = Cf.eval_model(
                model,
                val_dataloader_same,
                loss_fn,
                device,
                len(val_dataloader_same) * BATCH_SIZE,
                is_sigmoid
            )
            print(f'Val All = {val_acc_all} and SAME = {val_acc_same}')

        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if is_last_round and best_accuracy < val_acc:
            torch.save(model.state_dict(),
                       f'Models/Combined/4Way_Lenghth{LENGTH}_LEVEL{lvl}_epoch{epoch + 1}_model{val_acc}.bin')
            best_accuracy = val_acc

    test_acc, _ = Cf.eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(test_data_loader) * BATCH_SIZE,
        is_sigmoid
    )
    test_acc_all, _ = Cf.eval_model(
        model,
        test_dataloader_all,
        loss_fn,
        device,
        len(test_dataloader_all) * BATCH_SIZE,
        is_sigmoid
    )
    test_acc_same, _ = Cf.eval_model(
        model,
        test_dataloader_same,
        loss_fn,
        device,
        len(test_dataloader_same) * BATCH_SIZE,
        is_sigmoid
    )
    print("Test accuracy is: ", test_acc.item())
    print(f'Test acc = {test_acc}, Test ALL acc = {test_acc_all}, Test SAME acc = {test_acc_same}')
