import csv
import itertools
import heapq
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import Corefunctions as Cf
from transformers import BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The Device is:", device)

is_sigmoid = True
window_size = 4
poem_length = 10  # the length you want the poem to be generated.
starter_model_path = "Models/Micro/Starter SIGMOID 0.03 Lenghth_4 itter_400 all_0.8288352272727273 same_0.8359375.bin"
main_model_path = "Models/Combined/4Way_Lenghth4_LEVEL3_epoch99_model0.7798295454545455.bin"

with open('Data/examples2.csv', newline='') as f:
    reader = csv.reader(f)
    lines_lstlst2 = list(reader)

lines_lstlst = lines_lstlst2
top10_lstlst = [i[:10] for i in lines_lstlst]
top15_lstlst = [i[:15] for i in lines_lstlst]
top20_lstlst = [i[:20] for i in lines_lstlst]
top25_lstlst = [i[:25] for i in lines_lstlst]
top_lstlst = top25_lstlst

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 20 * 4
BATCH_SIZE = 200
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

starter_model = Cf.load_model(starter_model_path, PRE_TRAINED_MODEL_NAME, device, is_sigmoid)
main_model = Cf.load_model(main_model_path, PRE_TRAINED_MODEL_NAME, device, is_sigmoid)


def input_maker(lines_list: list, tokenizer):
    def encoderr(temp_lst):
        return tokenizer.encode_plus(
            temp_lst,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    temp = ""

    if isinstance(lines_list[0], str):
        temp = " [SEP] ".join(lines_list)  # if only one
        encoding = encoderr(temp)
    else:
        encoding = [encoderr(" [SEP] ".join(i)) for i in lines_list]  # if it is batched

    return encoding


class StartDataset(Dataset):

    def __init__(self, songs_lstlst, tokenizer, max_len):
        self.songs_lst = self.add_sep(songs_lstlst)  # list of poems that are seperated with [SEP]
        self.tokenizer = tokenizer
        self.max_len = max_len

    @staticmethod
    def add_sep(songs_lstlst):  # add [SEP] between lines in list of list of lines
        return [" [SEP] ".join(lines) for lines in songs_lstlst]

    def __len__(self):
        return len(self.songs_lst)

    def __getitem__(self, item):
        poem = str(self.songs_lst[item])

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
            'attention_mask': encoding['attention_mask'].flatten()
        }


def create_data_loader(songs_lstlst, tokenizer, max_len, batch_size):
    ds = StartDataset(
        songs_lstlst=np.array(songs_lstlst),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )


def best_start_suggestion(model, start_lines_n, first_idx, beam_size, is_sigmoid):
    """
    model: BERT model, if it is 3 liner or 4 liner or ...
    start_lines_n: number of lines that we want to start with, it should be suitable for the BERT model
    first_idx: fist index we want to start from in Olgas examples
    beam_size: beam search size. Will return top ? of the best
    """

    global top10_lstlst
    global device

    model = model.eval()
    start_lines_premutation = list(
        itertools.product(*top10_lstlst[first_idx:first_idx + start_lines_n]))  # All possible combinations
    dataloader = create_data_loader(start_lines_premutation, tokenizer, MAX_LEN, BATCH_SIZE)

    score_lst = []

    with torch.no_grad():
        for d in dataloader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            text = d["poem_text"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            scores = outputs.squeeze().tolist()
            if not is_sigmoid:
                scores = [i[1] for i in scores]
            score_lst.append(zip(scores, text))

    score_lst = itertools.chain.from_iterable(score_lst)  # this flattens list of Zip objects to list of tuples.
    score_lst = heapq.nlargest(beam_size, score_lst, key=lambda x: x[0])
    score_lst = [(i, j.split(" [SEP] ")) for i, j in score_lst]

    return score_lst


def predict_next(inps_lstlst, idx, beam_size, additive_score=True, is_sigmoid=False):
    """
    inps: list of the prevous lines wih their scores
    idx: index we want to pick the next line from in Olgas examples
    beam_size: beam search size. Will return top ? of the best
    """

    def inner_get_score(model, model_input_lenght, lines, next_line):
        temp_input = [*lines[-model_input_lenght:], next_line]  # e.g. -3 is for model4 input size
        temp_input = input_maker(temp_input, tokenizer)

        input_ids = temp_input["input_ids"].to(device)
        attention_mask = temp_input["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        score = outputs.squeeze().tolist()
        if not is_sigmoid:
            score = score[1]
        return score

    global main_model
    global top_lstlst
    global device

    results = []
    with torch.no_grad():
        for score, lines in inps_lstlst:
            for next_line in top_lstlst[idx]:
                new_score = inner_get_score(main_model, window_size - 1, lines, next_line)

                results.append([new_score + additive_score * score, [*lines, next_line]])

    score_lst = heapq.nlargest(beam_size, results, key=lambda x: x[0])
    return score_lst


def beam_search(start_idx, beam_size, beam_depth, additive_score=True):
    """
    start_idx: start index we want to start from in Olgas examples
    beam_size: beam search size. Will return top ? of the best
    beam_depth: beam search depth. How far we want to go.
    """

    global starter_model
    start_lines_n = window_size

    if beam_depth < start_lines_n:
        raise ValueError('Under 3 is not supporter for beam_depth')

    selected_lines = best_start_suggestion(starter_model, start_lines_n, start_idx, beam_size, is_sigmoid)

    for i in range(beam_depth - start_lines_n):
        idx_to_pick = start_idx + start_lines_n + i  # This is the index we should pick our lines from
        selected_lines = predict_next(selected_lines, idx_to_pick, beam_size, additive_score, is_sigmoid)

    return selected_lines


first_inx = 0  # fist index we want to start from in Olgas examples
start_lines_n = window_size  # number of lines that we want to start with

start_lines_premutation = list(itertools.product(*top_lstlst[first_inx:first_inx + start_lines_n]))

start_dataloader = create_data_loader(start_lines_premutation, tokenizer, MAX_LEN, BATCH_SIZE)

answers = []

for i in range(5):

    start_idx = random.randint(0, 450)

    temp4 = beam_search(start_idx, 1, poem_length, additive_score=False)
    temp4 = temp4[0][1]

    temp0 = [random.choice(i) for i in top_lstlst[start_idx:start_idx + poem_length]]

    temp_all = [(temp4, 5462464546246), (temp0, 5462461546246)]

    random.shuffle(temp_all)
    temps = [x[0] for x in temp_all]
    temp_labels = [x[1] for x in temp_all]
    answers.append(temp_labels)

    for temp in temps:
        for j in temp:
            print(j)
        print(" --------- ", i)
        print()

    print()
    print(" ================================================================== ")
    print()

print("*****************************")
print("*****************************")
print("*****************************")
print("*****************************")
for i in answers:
    print(*i)

print("*****************************")
print("*****************************")
print("*****************************")
print("*****************************")
