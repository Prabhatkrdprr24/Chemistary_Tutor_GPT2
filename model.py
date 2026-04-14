import torch
import torch.nn as nn 
import math


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer1 = nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4, bias=cfg["qkv_bias"])
        self.layer2 = nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_heads"]
        self.emb_dim = cfg["emb_dim"]
        self.h_dim = self.emb_dim // self.n_head

        assert (self.emb_dim % self.n_head == 0), "embedding dimension must be divisible by number of heads."

        self.q_w = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.k_w = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.v_w = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])

        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        batch, n_tokens, emb_dim = x.shape
        values = self.v_w(x)
        keys = self.k_w(x)
        queries = self.q_w(x)

        values = values.view(batch, n_tokens, self.n_head, self.h_dim)
        keys = keys.view(batch, n_tokens, self.n_head, self.h_dim)
        queries = queries.view(batch, n_tokens, self.n_head, self.h_dim)

        values = values.transpose(1, 2)    #(batch, n_head, n_tokens, h_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)     # (batch, n_head, n_tokens, n_tokens)

        mask = self.mask[:n_tokens, :n_tokens]

        attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.h_dim), dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values   # (batch, n_head, n_tokens, h_dim)

        context_vector = context_vector.transpose(1, 2)   # (batch, n_tokens, n_head, h_dim)

        context_vector = context_vector.contiguous().view(batch, n_tokens, self.emb_dim)

        return self.out(context_vector)
    


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)
        self.ff = FeedForward(cfg)
        self.mha = MultiHeadAttention(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)

        x = shortcut + x
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)

        x = shortcut + x
        return x
    


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

        self.trf = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm = LayerNorm(cfg)
        self.proj_out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # self.proj_out = nn.Linear(in_features=cfg["emb_dim"], out_features=cfg["num_classes"])

        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.scale)
            nn.init.zeros_(module.shift)

    def forward(self, input):
        batch, seq_len = input.shape
        tok_emb = self.tok_emb(input)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input.device))
        input_emb = tok_emb + pos_emb

        x = self.dropout(input_emb)

        x = self.trf(x)
        x = self.norm(x)
        logits = self.proj_out(x)

        return logits