import torch
from sklearn.model_selection import train_test_split
from transformers import BertModel, RobertaModel

import random
from tqdm import tqdm

import numpy as np

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def data_cleaner(songs):
    print("Number of songs: ", len(songs))
    songs = [i for i in songs if len(set(i)) > 15]
    print("Number of songs with more than 15 separate lines: ", len(songs))

    train_songs, test_songs = train_test_split(songs, random_state=RANDOM_SEED, test_size=0.10)
    val_songs, test_songs = train_test_split(test_songs, test_size=0.5, random_state=RANDOM_SEED)
    print(f'len train: {len(train_songs)}, len val: {len(val_songs)}, len test {len(test_songs)}')

    return train_songs, val_songs, test_songs


def mask_remove(string: str):
    tokens = string.split(" ")
    mask_number = len(tokens) // 4
    mask_indicis = random.sample(range(1, len(tokens)), mask_number)

    for index in sorted(mask_indicis, reverse=True):
        del tokens[index]

    return " ".join(tokens)


def mask_shuffle(string: str):
    tokens = string.split(" ")
    random.shuffle(tokens)
    return " ".join(tokens)


def _input_maker(songs_lst: list, length, level, version: str):
    """
    length: length of poem sequences (examples below)
    level:  number of true lines (examples below)
    version (str): There are five versions "same", "all", "start all" and "start same", "50/50".
                   if it is "all"  then the zero labels will be selected randomly from all the songs.
                   If it is "same" then the zero labels will be selected randomly from just that song.
                   If it is "start ..." then the true labels will be only selected from the start of the song
                   if it is "50/50" then it will be a 50% by 50% dataset of "same" and "all" together.
    ----
    length = 4, level = 2: [True, Ture, Random, Random]
    length = 3, level = 0: [Random, Random, Random]
    length = 6, level = 2: [True, Ture, Random, Random, Random, Random]
    length = 6, level = 5: [True, Ture, True, Ture, True, Random]
    """

    X = []
    Y = []

    for song in songs_lst:
        for _ in range(1):  # Just one sample from a song
            if "start" in version:
                rnd_temp = 0  # starting index
                assert level == 0, f"for Start datasets level must be zero but here level is {level}"
            else:
                rnd_temp = random.randint(0, len(song) - length - 1)

            selected_lines = song[rnd_temp:rnd_temp + length]

            if random.random() > 0.5:
                X.append(" [SEP] ".join(selected_lines))
                Y.append(1)
            else:
                kept_lines = selected_lines[:level]
                masked_lines = selected_lines[level:]

                if "all" in version:
                    bag_of_lines = random.choice(songs_lst)  # A random song from all songs
                elif "same" in version:
                    bag_of_lines = tuple(set(song).difference(masked_lines))  # Remove the true lines from the bag
                elif "50/50" in version:
                    rand = random.random()
                    if rand < 0.25:
                        bag_of_lines = random.choice(songs_lst)  # A random song from all songs
                    elif rand < 0.5:
                        bag_of_lines = tuple(set(song).difference(masked_lines))
                    elif rand < 0.75:
                        bag_of_lines = [mask_remove(i) for i in masked_lines]
                    else:
                        bag_of_lines = [mask_shuffle(i) for i in masked_lines]
                else:
                    raise ValueError("version most be only 'all', 'same', 'start all' and 'start same' or '50/50'")
                kept_lines.extend([random.choice(bag_of_lines) for i in range(length - level)])

                X.append(" [SEP] ".join(kept_lines))
                Y.append(0)

    return X, Y


class GPReviewDataset(Dataset):

    def __init__(self, poems, targets, tokenizer, max_len):
        self.poems = poems
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, item):
        poem = str(self.poems[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            poem,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'poem_text': poem,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def _create_data_loader(poems_lst, tokenizer, max_len, batch_size, length, level, version: str):
    inputs, targets = _input_maker(poems_lst, length, level, version)

    ds = GPReviewDataset(
        poems=np.array(inputs),
        targets=np.array(targets),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )


def create_dataloaders(train_songs, val_songs, test_songs, tokenizer, batch_size, max_len, length, level, version: str):
    # creates test, val, test data loaders
    """
    length: length of poem sequences (examples below)
    level:  number of true lines (examples below)
    version (str): There are five versions "same", "all", "start all" and "start same", "50/50".
                   if it is "all"  then the zero labels will be selected randomly from all the songs.
                   If it is "same" then the zero labels will be selected randomly from just that song.
                   If it is "start ..." then the true labels will be only selected from the start of the song
                   if it is "50/50" then it will be a 50% by 50% dataset of "same" and "all" together.
    ----
    length = 4, level = 2: [True, Ture, Random, Random]
    length = 3, level = 0: [Random, Random, Random]
    length = 6, level = 2: [True, Ture, Random, Random, Random, Random]
    length = 6, level = 5: [True, Ture, True, Ture, True, Random]
    """
    train_data_loader = _create_data_loader(train_songs, tokenizer, max_len, batch_size, length, level, version)
    val_data_loader = _create_data_loader(val_songs, tokenizer, max_len, batch_size, length, level, version)
    test_data_loader = _create_data_loader(test_songs, tokenizer, max_len, batch_size, length, level, version)

    return train_data_loader, val_data_loader, test_data_loader


###################################################################################################

def _input_maker_test(songs_lst: list, length, level, version: str):
    """
    length: length of poem sequences (examples below)
    level:  number of true lines (examples below)
    version (str): There are five versions "same", "all", "start all" and "start same", "50/50".
                   if it is "all"  then the zero labels will be selected randomly from all the songs.
                   If it is "same" then the zero labels will be selected randomly from just that song.
                   If it is "start ..." then the true labels will be only selected from the start of the song
                   if it is "50/50" then it will be a 50% by 50% dataset of "same" and "all" together.
    ----
    length = 4, level = 2: [True, Ture, Random, Random]
    length = 3, level = 0: [Random, Random, Random]
    length = 6, level = 2: [True, Ture, Random, Random, Random, Random]
    length = 6, level = 5: [True, Ture, True, Ture, True, Random]
    """

    X = []
    Y = []

    for song in songs_lst:
        for _ in range(1):  # Just one sample from a song

            rnd_temp = random.randint(0, len(song) - length - 1)

            selected_lines = song[rnd_temp:rnd_temp + length]

            if random.random() > 0.5:
                X.append(" [SEP] ".join(selected_lines))
                Y.append(1)
            else:
                kept_lines = selected_lines[:level]
                masked_lines = selected_lines[level:]
                if random.random() > 0.5:  # all
                    kept_lines.extend([random.choice(random.choice(songs_lst)) for i in range(length - level)])
                else:  # same
                    selected_lines_complement = tuple(set(song).difference(masked_lines))
                    kept_lines.extend([random.choice(selected_lines_complement) for i in range(length - level)])
                X.append(" [SEP] ".join(kept_lines))
                Y.append(0)

    return X, Y


def _create_data_loader_test(poems_lst, tokenizer, max_len, batch_size, length, level, version: str):
    inputs, targets = _input_maker_test(poems_lst, length, level, version)

    ds = GPReviewDataset(
        poems=np.array(inputs),
        targets=np.array(targets),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )


def test_create_dataloaders(train_songs, val_songs, test_songs, tokenizer, batch_size, max_len, length, level,
                            version: str):
    train_data_loader = _create_data_loader_test(train_songs, tokenizer, max_len, batch_size, length, level, version)
    val_data_loader = _create_data_loader_test(val_songs, tokenizer, max_len, batch_size, length, level, version)
    test_data_loader = _create_data_loader_test(test_songs, tokenizer, max_len, batch_size, length, level, version)

    return train_data_loader, val_data_loader, test_data_loader


###################################################################################################


class CohClassifier(nn.Module):

    def __init__(self, n_classes, pre_trained_model_name):
        super(CohClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class CohClassifierSigmoid(nn.Module):  # The last layer has sigmoid

    def __init__(self, pre_trained_model_name):
        super(CohClassifierSigmoid, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        output = torch.flatten(output)
        return self.sigmoid(output)


###################################################################################################


def loss_and_acc(model, batch_of_data, loss_fn, device, is_sigmoid):  # return loss and accuracy for a batch
    input_ids = batch_of_data["input_ids"].to(device)
    attention_mask = batch_of_data["attention_mask"].to(device)
    targets = batch_of_data["targets"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    if is_sigmoid:
        loss = loss_fn(outputs, targets.float())
        predictions = (outputs > 0.5).float()
    else:
        loss = loss_fn(outputs, targets)
        _, predictions = torch.max(outputs, dim=1)

    correct_predictions_in_batch = torch.sum(predictions == targets)

    return loss, correct_predictions_in_batch


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, is_sigmoid=False):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        loss, correct_predictions_in_batch = loss_and_acc(model, d, loss_fn, device, is_sigmoid)

        correct_predictions += correct_predictions_in_batch
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples, is_sigmoid=False):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            loss, correct_predictions_in_batch = loss_and_acc(model, d, loss_fn, device, is_sigmoid)

            correct_predictions += correct_predictions_in_batch
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def load_model(path: str, pre_trained_model_name, device, is_sigmoid):
    if is_sigmoid:
        model = CohClassifierSigmoid(pre_trained_model_name)
    else:
        model = CohClassifier(2, pre_trained_model_name)  # true 1, false 0
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model = model.eval()
    return model
