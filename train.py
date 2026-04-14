import torch
from tqdm import tqdm
from pathlib import Path



from load_weights import load_weights_into_gpt
from model import Model
from config import cfg
from utils import get_weight_file, tokenizer, generate, text_to_ids, ids_to_text, download_and_load_gpt2
from dataset import format_input, train_loader, val_loader, val_data, format_for_generate

model = Model(cfg)

settings, params = download_and_load_gpt2(
    model_size="335M",
    models_dir="gpt2-335M"
)

load_weights_into_gpt(model, params)
# model.eval()



def batch_loss(model, target_batch, input_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def loader_loss(model, loader, device, num_batch):
    total_loss = 0.0
    if len(loader) == 0:
        return float("nan")
    if num_batch is None:
        num_batch = len(loader)
    else:
        num_batch = min(num_batch, len(loader))
    
    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batch:
            loss = batch_loss(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batch

def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.inference_mode():
        train_loss = loader_loss(model=model, loader=train_loader, device=device, num_batch=eval_iter)
        val_loss = loader_loss(model=model, loader=val_loader, device=device, num_batch=eval_iter)
    # model.train()   # single responsiblity principle
    return train_loss, val_loss


def train(model, train_loader, val_loader, cfg, tokenizer, val_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device -> {device}")

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
            initial_step = state.get("initial_epoch", 0)
            global_step = state.get("global_step", -1)
            print(f"Preloaded weights from {weight_filename}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print("no weights found.")

    for epoch in range(initial_step, cfg["epochs"]):
        model.train()

        batch_iterator = tqdm(train_loader, desc='processing batch')
        for input_batch, target_batch in batch_iterator:
            optimizer.zero_grad()
            loss = batch_loss(input_batch=input_batch, target_batch=target_batch, device=device, model=model)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % cfg["eval_freq"] == 0:
                train_loss, val_loss = evaluate(device=device, eval_iter=cfg["eval_iter"], model=model, train_loader=train_loader, val_loader=val_loader)
                model.train()
                print(f"epoch: {epoch} (global step: {global_step:06d})"
                      f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")
                
        # generated_text = ids_to_text(ids=generate(cfg=cfg, device=device, eos=50256, model=model, ids=text_to_ids(text=format_input(entry=val_data[0]), tokenizer=tokenizer), max_new_tokens=cfg["max_new_tokens"]), tokenizer=tokenizer)
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
            "global_step": global_step,
            "initial_epoch": epoch + 1
        }, weights)

        print("model weights and optimizer saved.")
                
            

train(cfg=cfg, model=model, tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader, val_data=val_data)