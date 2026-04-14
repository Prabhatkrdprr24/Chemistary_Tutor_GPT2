import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from load_weights import load_weights_into_gpt
from model import Model
from config import cfg
from utils import get_weight_file, tokenizer, generate, text_to_ids, ids_to_text, download_and_load_gpt2
from dataset import format_input, train_loader, val_loader, val_data, format_for_generate


model = Model(cfg)

settings, params = download_and_load_gpt2(
    model_size="355M",
    models_dir="gpt2-255M"
)

load_weights_into_gpt(model, params)


def batch_loss(model, target_batch, input_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def loader_loss(model, loader, device, num_batch, use_amp):
    total_loss = 0.0
    if len(loader) == 0:
        return float("nan")
    if num_batch is None:
        num_batch = len(loader)
    else:
        num_batch = min(num_batch, len(loader))
    
    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batch:
            with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = batch_loss(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batch

def evaluate(model, train_loader, val_loader, device, eval_iter, use_amp):
    model.eval()
    with torch.inference_mode():
        train_loss = loader_loss(model=model, loader=train_loader, device=device, num_batch=eval_iter, use_amp=use_amp)
        val_loss = loader_loss(model=model, loader=val_loader, device=device, num_batch=eval_iter, use_amp=use_amp)
    # model.train()   # single responsiblity principle
    return train_loss, val_loss

def train(model, train_loader, val_loader, cfg, tokenizer, val_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device -> {device}")

    use_amp = (device == "cuda")

    scaler = GradScaler(device=device, enabled=use_amp)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    initial_step, global_step = 0, -1

    model.to(device)

    if cfg["preload"]:
        weight_filename = get_weight_file(cfg, cfg["preload"])
        print("pre-loading weights -> ", weight_filename)
        try:
            state = torch.load(weight_filename, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scaler_state_dict" in state:
                scaler.load_state_dict(state["scaler_state_dict"])
            initial_step = state.get("initial_epoch", 0)
            global_step = state.get("global_step", -1)
            print(f"Preloaded weights from {weight_filename}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print("no weights found.")

    for epoch in range(initial_step, cfg["epochs"]):
        model.train()

        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch} processing batch")
        for input_batch, target_batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = batch_loss(input_batch=input_batch, target_batch=target_batch, device=device, model=model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if global_step % cfg["eval_freq"] == 0:
                train_loss, val_loss = evaluate(device=device, eval_iter=cfg["eval_iter"], model=model, train_loader=train_loader, val_loader=val_loader, use_amp=use_amp)
                model.train()
                print(f"epoch: {epoch} (global step: {global_step:06d}) "
                      f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

        ids = text_to_ids(text=format_for_generate(text=""), tokenizer=tokenizer)
        print(ids_to_text(ids=ids, tokenizer=tokenizer))
        generated_text = generate(cfg=cfg, device=device, eos=50256, max_new_tokens=cfg["max_new_tokens"], model=model, ids=ids, tokenizer=tokenizer)

        for token in generated_text:
            print(token, end="", flush=True)

        # generated_text = ids_to_text(ids=generate(cfg=cfg, device=device, eos=50256, model=model, ids=text_to_ids(text=format_for_generate(text="Which layer of soil experiences the most weathering?"), tokenizer=tokenizer), max_new_tokens=cfg["max_new_tokens"]), tokenizer=tokenizer)
        # print(f"Generated text: {generated_text}")
        # print(f"Length of generated text: {len(generated_text)}")
        # print(f"Generated text: {generated_text}")

        weights = get_weight_file(cfg, epoch=epoch)

        torch.save({
            "model_state_dict": model.state_dict()
        }, f"weights_only_{epoch}.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": global_step,
            "initial_epoch": epoch + 1
        }, weights)

        print("model weights and optimizer saved.")