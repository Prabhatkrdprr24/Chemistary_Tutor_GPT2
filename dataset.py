import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import urllib
import ssl
from config import cfg
from functools import partial

from utils import tokenizer

def download_and_load_file(file_path, url):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as res:
            text_data = res.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

# file_path = ".data.json"
# url = cfg["data_url"]

# data = download_and_load_file(file_path=file_path, url=url)
# print("data downloaded. length of data: ", len(data))

# train_ratio = int(len(data) * 0.95)
# val_ratio = int(len(data) * 0.025)
# test_ratio = int(len(data) * 0.025)

# train_data = data[:train_ratio]
# val_data = data[train_ratio:train_ratio+val_ratio]
# test_data = data[train_ratio+val_ratio:]

# print(f"Length of train data -> {len(train_data)}")
# print(f"Length of val data -> {len(val_data)}")
# print(f"Length of test data -> {len(test_data)}")

def format_input(entry):
    instruction_text = (
        f"You are a helpful and friendly chemistry tutor. Answer the student's question clearly and concisely. Give examples if needed. Encourage further thinking if possible."
        f"\n\n### Question:\n{entry['Input']}"
    )
    response_text = f"\n\n### Answer:\n{entry['Output']}"
    return instruction_text + response_text

def format_for_generate(text):
    instruction_text = (
        f"You are a helpful and friendly chemistry tutor. Answer the student's question clearly and concisely. Give examples if needed. Encourage further thinking if possible."
        f"\n\n### Question:\n{text}"
    )
    respones_text = f"\n\n### Answer:\n"
    return instruction_text + respones_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data

        self.encoded = []
        for entry in data:
            input = format_input(entry)
            self.encoded.append(
                tokenizer.encode(input)
            )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded[index]
    

def custom_collate_fn(batch, device, ignore_index=-100, allowed_max_length=None, pad_id=50256):

    max_length = max(len(item)+1 for item in batch)
    input_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_id]
        padded = (
            new_item + [pad_id] * (max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            allowed_inputs = inputs[:allowed_max_length]
            allowed_targets = targets[:allowed_max_length]

        input_lst.append(allowed_inputs)
        target_lst.append(allowed_targets)

    input_tensor = torch.stack(input_lst).to(device)
    target_tensor = torch.stack(target_lst).to(device)

    return input_tensor, target_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=cfg["context_length"])

# train_dataset = InstructionDataset(data=train_data, tokenizer=tokenizer)
# train_loader = DataLoader(
#     train_dataset,
#     collate_fn=customized_collate_fn,
#     batch_size=cfg["batch_size"],
#     drop_last=True,
#     shuffle=True,
#     num_workers=cfg["num_workers"]
# )

# test_dataset = InstructionDataset(data=test_data, tokenizer=tokenizer)
# test_loader = DataLoader(
#     test_dataset,
#     collate_fn=customized_collate_fn,
#     batch_size=cfg["batch_size"],
#     drop_last=False,
#     shuffle=False,
#     num_workers=cfg["num_workers"]
# )

# val_dataset = InstructionDataset(data=val_data, tokenizer=tokenizer)
# val_loader = DataLoader(
#     val_dataset,
#     collate_fn=customized_collate_fn,
#     batch_size=cfg["batch_size"],
#     drop_last=False,
#     shuffle=False,
#     num_workers=cfg["num_workers"]
# )



