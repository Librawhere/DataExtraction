import json

import torch
from tqdm import tqdm
from model import Pythia
from transformers import AutoTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Pythia()
model.load_state_dict(torch.load('weights/adatr70.pt'))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

total = 0
ht = 0
test_ds = []
flag = 0

with open('dataset/pt_inference.json', 'r', encoding='utf-8') as f:
    ds = json.load(f)
    for d in ds:
        test_ds.append(d)

for data in tqdm(test_ds):
    inputs = tokenizer(data[0], return_tensors="pt").to(device)
    tokens = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(tokens[0], skip_special_tokens=True)

    # # greedy decoding evaluation
    # # soft rule
    # items = data[1].split(' ')
    # for item in items:
    #     if item in pred:
    #         ht += 1
    #         break
    #
    # # hard rule
    # # if data[1] in pred:
    # #     ht += 1
    # total += 1

    # beam search evaluation
    total += 1
    for i, beam_output in enumerate(tokens):
        pred = tokenizer.decode(beam_output, skip_special_tokens=True)

        # soft rule
        items = data[1].split(' ')
        for item in items:
            if item in pred:
                ht += 1
                flag = 1
                break
        if flag == 1:
            flag = 0
            break

    # hard rule
    #     if data[1] in pred:
    #         ht += 1
    #         break


leakage = ht/total

print(f'Leakage is {leakage}')








