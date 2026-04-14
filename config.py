

# cfg = {
#     "emb_dim": 1600,
#     "vocab_size": 50257,
#     "qkv_bias": True,
#     "n_heads": 25,
#     "context_length": 1024,
#     "dropout": 0.1,
#     "n_layers": 48,
#     "preload": None,
#     "weight_folder": "weights",
#     "weight_basename": "_tmodel",
#     "weight_decay": 0.1,
#     "learning_rate": 5e-5,
#     "epochs": 5,
#     "eval_freq": 20,
#     "eval_iter": 10,
#     "num_classes": 2,
#     "batch_size": 15,
#     "num_workers": 0,
#     "max_new_tokens": 100,
    

# }




cfg = {
    "emb_dim": 1024,
    "vocab_size": 50257,
    "qkv_bias": True,
    "n_heads": 16,
    "context_length": 1024,
    "dropout": 0.1,
    "n_layers": 24,
    "preload": 2,
    "weight_folder": "weights",
    "weight_basename": "_tmodel",
    "weight_decay": 0.1,
    "learning_rate": 5e-5,
    "epochs": 5,
    "eval_freq": 50,
    "eval_iter": 10,
    "num_classes": 2,
    "batch_size": 1,
    "num_workers": 0,
    "max_new_tokens": 200,
    

}