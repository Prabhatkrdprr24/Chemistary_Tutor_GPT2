import torch
import tiktoken
import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(encoded).unsqueeze(dim=0)
    return ids

def ids_to_text(ids, tokenizer):
    ids = ids.squeeze(0)
    return tokenizer.decode(ids.tolist())


def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)


    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True, verify=False)

        file_size = int(response.headers.get("content-length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        block_size = 1024

        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def get_weight_file(cfg, epoch):
    weight_folder = cfg["weight_folder"]
    weight_basename = cfg["weight_basename"]
    weight_filename = f"{weight_basename}{epoch}.pt"
    return str(Path(".")/weight_folder/weight_filename)


# def generate(model, ids,  cfg, device, max_new_tokens, temperature=0.0, top_k=None, eos=50256):
#     model.eval()

#     for _ in range(max_new_tokens):
#         ids = ids.to(device)
#         ids_cond = ids[:, -cfg["context_length"]:]
#         with torch.inference_mode():
#             logits = model(ids_cond)     # model output shape -> (batch, context_length, vocab_size)
#         logits = logits[:, -1, :]

#         if top_k is not None:
#             top_logits, _ = torch.topk(logits, top_k)
#             min_val = top_logits[:, -1]
#             logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

#         if temperature > 0.0:
#             logits = logits / temperature
#             probas = torch.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probas, num_samples=1)
#         else:
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)

#         if eos is not None:
#             if(idx_next == eos).all():
#                 break

#         ids = torch.cat((ids, idx_next), dim=-1)

#     return ids


def generate(model, ids,  cfg, device, tokenizer, max_new_tokens, temperature=0.0, top_k=None, eos=50256):
    model.eval()

    for _ in range(max_new_tokens):
        ids = ids.to(device)
        ids_cond = ids[:, -cfg["context_length"]:]
        with torch.inference_mode():
            logits = model(ids_cond)     # model output shape -> (batch, context_length, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos is not None:
            if(idx_next == eos).all():
                break

        ids = torch.cat((ids, idx_next), dim=-1)

        yield tokenizer.decode(idx_next[0].tolist())

    # return ids
