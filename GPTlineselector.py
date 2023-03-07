import heapq
import itertools

import numpy as np
from aitextgen import aitextgen
import Corefunctions as Cf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The Device is:", device)

is_sigmoid = True
window_size = 4
poem_length = 10  # the length you want the poem to be generated.
starter_model_path = "Models/Micro/Starter SIGMOID 0.03 Lenghth_4 itter_400 all_0.8288352272727273 same_0.8359375.bin"
main_model_path = "Models/Combined/4Way_Lenghth4_LEVEL3_epoch99_model0.7798295454545455.bin"

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 20 * 4
BATCH_SIZE = 200
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

####################################LOADING MODELS ####################################
ai = aitextgen(model_folder='.', to_gpu=True)
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


def best_start_suggestion(model, songs_lstlst, beam_size, is_sigmoid):
    """
    model: BERT model, if it is 3 liner or 4 liner or ...
    start_lines_n: number of lines that we want to start with, it should be suitable for the BERT model
    first_idx: fist index we want to start from in Olgas examples
    beam_size: beam search size. Will return top ? of the best
    """
    global device

    model = model.eval()

    dataloader = create_data_loader(songs_lstlst, tokenizer, MAX_LEN, BATCH_SIZE)

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
    score_lst = [(score, text.split(" [SEP] ")) for score, text in score_lst]

    return score_lst


def starter(ai, prompt, number, beam_size):
    """
    :param ai: the model GPT
    :param prompt:
    :param number: number of generation to be picked out of. the more the better.
    :param beam_size:
    :return:
    """
    outputs = []

    for i in range(number):
        gen = ai.generate_one(prompt=prompt,
                              max_length=100,
                              temperature=0.9,
                              top_p=0.9)
        outputs.append(gen)

    outputs = [i.split("#") for i in outputs]
    outputs = [[j.strip() for j in i if j] for i in outputs]  # getting rid on any empty
    outputs = [i for i in outputs if len(i) >= 4]
    outputs = [i[:4] for i in outputs]

    return best_start_suggestion(starter_model, outputs, beam_size, is_sigmoid)


def predict_next(inps_lstlst, next_lines, beam_size, additive_score=True, is_sigmoid=False):
    """
    inps: list of the prevous lines wih their scores
    idx: index we want to pick the next line from in Olgas examples
    beam_size: beam search size. Will return top ? of the best
    """

    def inner_get_score(model, model_input_length, lines, next_line):
        temp_input = [*lines[-model_input_length:], next_line]  # e.g. -3 is for model4 input size
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
    global device

    results = []
    with torch.no_grad():
        for score, lines in inps_lstlst:
            for next_line in next_lines:
                new_score = inner_get_score(main_model, window_size - 1, lines, next_line)
                results.append([new_score + additive_score * score, [*lines, next_line]])

    score_lst = heapq.nlargest(beam_size, results, key=lambda x: x[0])
    return score_lst


def greedy(ai, prompt, depth, number):
    selected_lines = starter(ai, prompt, depth, 1)  # this is with the scores (score, lines)
    previous_lines = selected_lines[0][1]

    for d in range(depth):
        next_lines = []
        prompt = " # ".join(previous_lines) + " #"
        max_length = (5 + d) * 10

        for i in range(number):
            gen = ai.generate_one(prompt=prompt,
                                  max_length=max_length,
                                  temperature=0.9,
                                  top_p=0.9)
            gen = gen[len(prompt):].split(" # ")[0].strip()
            next_lines.append(gen)

        selected_lines = predict_next(selected_lines, next_lines, beam_size=1, is_sigmoid=is_sigmoid)
        previous_lines = selected_lines[0][1]

    return selected_lines


def beam_search(ai, prompt, beam_depth, beam_size, number, additive_score=True):
    """
    :param ai: the model
    :param prompt:
    :param beam_depth: How many lines we want to generate? How long our poem we want to be?
    :param beam_size: How many beams we want to keep in each iteration?
    :param number: Number of generation to be picked out of. the more the better.
    :param additive_score:
    :return:
    """
    selected_lines = starter(ai, prompt, number, beam_size)  # this is with the scores (score, lines)
    previous_lines = [i[1] for i in selected_lines]

    for d in range(beam_depth - 4):  # Because we already have 4 starting lines.
        next_lines = set()
        max_length = (5 + d) * 10
        for branch in previous_lines:
            prompt = " # ".join(branch) + " #"
            for i in range(number):
                gen = ai.generate_one(prompt=prompt,
                                      max_length=max_length,
                                      temperature=0.9,
                                      top_p=0.9)
                gen = gen[len(prompt):].split(" # ")[0].strip()
                if len(gen) > 3:
                    next_lines.add(gen)

        selected_lines = predict_next(selected_lines, next_lines, beam_size, is_sigmoid=is_sigmoid)
        previous_lines = [i[1] for i in selected_lines]

    return selected_lines[0]


def printer(prompt):
    prompt = prompt + " #"
    print("##################################################")
    print()

    result = beam_search(ai, prompt, 10, 4, 8)
    for i in result[1]:
        print(i)

    print("----------------")

    gen = ai.generate_one(prompt=prompt,
                          max_length=250,
                          temperature=0.9,
                          top_p=0.9)
    gen = gen.split(" # ")[:10]
    for i in gen:
        print(i)
    print()


printer("Sometimes, all I think about is you")
printer("Can you kiss me more?")
printer("I do the same thing, I told you that I never would")
printer("I don't know what you've been told")
printer("I got my driver's license last week")
