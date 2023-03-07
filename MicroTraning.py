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

############################################################################################# PARAMETERS

is_sigmoid = True  # there are two models, with sigmoid and without sigmoid.

portion = 0.03
is_cascade = False
LENGTH = 4
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
BATCH_SIZE = 64
EPOCHS = 1

PATH_TO_SONGS = "Data/songs_data.csv"
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

#############################################################################################

print("Loading the songs file..........")
with open(PATH_TO_SONGS, newline='') as f:
    songs = list(csv.reader(f))

all_train_songs, val_songs, test_songs = Cf.data_cleaner(songs)

if is_sigmoid:
    model = Cf.CohClassifierSigmoid(PRE_TRAINED_MODEL_NAME)  # true 1, false 0
else:
    model = Cf.CohClassifier(2, PRE_TRAINED_MODEL_NAME)  # true 1, false 0

if PATH_TO_PRETRAINED_MODEL:
    model.load_state_dict(torch.load(PATH_TO_PRETRAINED_MODEL))
model = model.to(device)

#############################################################################################

# loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = nn.BCELoss().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

lvl = LENGTH - 1
batch_size = BATCH_SIZE
num_epochs = EPOCHS  # the last one has one more epoch to be trained

_, val_dl_all, test_dl_all = Cf.create_dataloaders(all_train_songs, val_songs, test_songs, tokenizer, batch_size,
                                                   MAX_LEN, LENGTH, lvl, version="all")
_, val_dl_same, test_dl_same = Cf.create_dataloaders(all_train_songs, val_songs, test_songs, tokenizer, batch_size,
                                                     MAX_LEN, LENGTH, lvl, version="same")

num_itter = 10000000
summation = 0

for itter in range(num_itter):
    version = "same" if itter % 2 else "all"
    val_data_loader = val_dl_same if itter % 2 else val_dl_all

    print(f" ********* Epoch {itter + 1}/{num_itter} length {LENGTH} *********")
    if itter > 300 or not itter % 100:  # todo: this can be changed
        test_acc_all, _ = Cf.eval_model(  # sigmoid
            model,
            test_dl_all,
            loss_fn,
            device,
            len(test_dl_all) * batch_size,
            is_sigmoid=is_sigmoid
        )
        print("ALL Test accuracy is: ", test_acc_all.item())
        test_acc_same, _ = Cf.eval_model(
            model,
            test_dl_same,
            loss_fn,
            device,
            len(test_dl_same) * batch_size,
            is_sigmoid=is_sigmoid
        )

        print("SAME Test accuracy is: ", test_acc_same.item())
        with open(f'SIGMOID Log {portion}n_L{LENGTH}.log', "a") as f:
            f.write(f'\n{itter}\t{test_acc_all}\t{test_acc_same}')

        if test_acc_same + test_acc_all > summation + 0.002:
            summation = test_acc_all + test_acc_same
            print("NEWWWWWWWWWWWWWW VAL:  ", summation)
            try:
                torch.save(model.state_dict(),
                           f'Models/Micro/SIGMOID {portion} Lenghth_{LENGTH} itter_{itter} all_{test_acc_all} same_{test_acc_same}.bin')
            except:
                print("ERORRRRRRRRRRRRRRRR")

    _, train_songs = train_test_split(all_train_songs, test_size=portion)
    train_data_loader = Cf._create_data_loader(train_songs, tokenizer, MAX_LEN, batch_size, LENGTH, lvl, version)

    total_steps = len(train_data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(num_epochs):  # + 1 if it is the last lvl
        print('-' * 10)

        train_acc, train_loss = Cf.train_epoch(  # sigmoid
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_data_loader) * batch_size,
            is_sigmoid=is_sigmoid
        )

        print(f'\n {version} Train loss {train_loss} accuracy {train_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
