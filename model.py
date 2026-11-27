"""Tiny GPT-style decoder written from scratch (no pretrained weights)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict

@dataclass
class Config:
    vocab_size: int = 8192
    ctx: int = 640
    d: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.0

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln1 = nn.LayerNorm(c.d); s.ln2 = nn.LayerNorm(c.d)
        s.qkv = nn.Linear(c.d, 3 * c.d, bias=False); s.proj = nn.Linear(c.d, c.d, bias=False)
        s.fc = nn.Linear(c.d, 4 * c.d, bias=False); s.fc2 = nn.Linear(4 * c.d, c.d, bias=False)
        s.nh = c.n_head; s.drop = c.dropout
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, s.nh, C // s.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.drop if s.training else 0)
        x = x + s.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + s.fc2(F.gelu(s.fc(s.ln2(x))))

class TinyGPT(nn.Module):
    def __init__(s, c: Config):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c.vocab_size, c.d); s.pos = nn.Embedding(c.ctx, c.d)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        s.ln = nn.LayerNorm(c.d)
        s.head = nn.Linear(c.d, c.vocab_size, bias=False)
        s.head.weight = s.tok.weight  # weight tying
        s.apply(s._init)
        for n, p in s.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
    def forward(s, idx, targets=None, mask=None):
        T = idx.shape[1]
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks: x = b(x)
        logits = s.head(s.ln(x))
        if targets is None: return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if mask is not None:
            m = mask.reshape(-1).float(); loss = (loss * m).sum() / m.sum().clamp(min=1)
        else: loss = loss.mean()
        return logits, loss
    def n_params(s): return sum(p.numel() for p in s.parameters()) - s.pos.weight.numel()


# ---------------------------------------------------------------- inference with a KV cache
def _block_cached(b, x, kv, pos0):
    B, T, C = x.shape
    q, k, v = b.qkv(b.ln1(x)).split(C, dim=2)
    q, k, v = (t.view(B, T, b.nh, C // b.nh).transpose(1, 2) for t in (q, k, v))
    if kv is not None:
        k = torch.cat([kv[0], k], dim=2); v = torch.cat([kv[1], v], dim=2)
    S = k.shape[2]
    mask = torch.arange(S, device=x.device)[None, :] <= (torch.arange(T, device=x.device)[:, None] + pos0)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + b.proj(y.transpose(1, 2).reshape(B, T, C))
    return x + b.fc2(F.gelu(b.fc(b.ln2(x)))), (k, v)

@torch.no_grad()
def forward_cached(model, idx, cache, pos0):
    """idx: (1,T) new tokens; cache: list per layer (or None). Returns logits of last token and new cache."""
    T = idx.shape[1]
    x = model.tok(idx) + model.pos(torch.arange(pos0, pos0 + T, device=idx.device))
    new = []
    for i, b in enumerate(model.blocks):
        x, kv = _block_cached(b, x, cache[i] if cache else None, pos0)
        new.append(kv)
    return model.head(model.ln(x[:, -1:]))[:, -1], new
