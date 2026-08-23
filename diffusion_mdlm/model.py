import math

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
mask_id = vocab_size
decode = lambda l: enc.decode([t for t in l if t < vocab_size])

block_size = 128


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def topk_sample(logits, k=20):
    v = logits.topk(min(k, logits.size(-1)), dim=-1).values
    logits = logits.masked_fill(logits < v[..., -1:], float('-inf'))
    return torch.multinomial(F.softmax(logits, -1).view(-1, vocab_size), 1).view(logits.shape[:-1])


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, drop_rate):
        super().__init__()
        self.n_heads = num_heads
        self.n_embed = n_embed
        self.drop_rate = drop_rate
        head_dim = n_embed // num_heads
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.dropout = nn.Dropout(drop_rate)
        half = head_dim // 2
        freqs = 1.0 / (1000 ** (torch.arange(half).float() / half))
        ang = torch.outer(torch.arange(block_size).float(), freqs)
        self.register_buffer('rope_cos', torch.cos(ang), persistent=False)
        self.register_buffer('rope_sin', torch.sin(ang), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embed, dim=2)
        q = self.q_norm(q.view(B, T, self.n_heads, C // self.n_heads)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.n_heads, C // self.n_heads)).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        cos, sin = self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        drop = self.drop_rate if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, drop_rate):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, num_heads, block_size, drop_rate)
        self.ffwd = FeedForward(n_embed, drop_rate)
        self.ln1 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_embed, elementwise_affine=False)
        self.adaLN = nn.Linear(n_embed, 4 * n_embed)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, temb):
        s1, c1, s2, c2 = self.adaLN(F.silu(temb))[:, None, :].chunk(4, dim=-1)
        x = x + self.sa(self.ln1(x) * (1 + c1) + s1)
        x = x + self.ffwd(self.ln2(x) * (1 + c2) + s2)
        return x


class TextDiffusion(nn.Module):
    def __init__(self, n_embed=512, n_layers=6, n_heads=16, block_size=128, drop_rate=0.0):
        super().__init__()
        self.n_embed = n_embed
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size + 1, n_embed)
        self.blocks = nn.ModuleList([
            Block(n_embed, n_heads, block_size, drop_rate) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.SiLU(),
            nn.Linear(n_embed, n_embed),
        )

    def forward(self, idx, t):
        temb = self.time_mlp(timestep_embedding(t, self.n_embed))
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x, temb)
        return self.lm_head(self.ln(x))

    @torch.no_grad()
    def generate(self, n_samples, steps=128, k=20):
        device = next(self.parameters()).device
        x = torch.full((n_samples, self.block_size), mask_id, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(n_samples)
            x0_hat = topk_sample(self(x, t), k)
            is_mask = x == mask_id
            unmask_p = (ts[i] - ts[i + 1]) / ts[i].clamp_min(1e-6)
            do = is_mask & (torch.rand(x.shape, device=device) < unmask_p)
            x = torch.where(do, x0_hat, x)
        is_mask = x == mask_id
        if is_mask.any():
            x0_hat = topk_sample(self(x, ts[-1].expand(n_samples)), k)
            x = torch.where(is_mask, x0_hat, x)
        return x
