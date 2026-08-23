import math

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
decode = lambda l: enc.decode(l)

block_size = 128
lam_min, lam_max = 1e-3, 8.0


def noise(t):
    lam = lam_min * (lam_max / lam_min) ** t
    log_ratio = math.log(lam_max / lam_min)
    return lam, lam * log_ratio


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
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
    def generate(self, n_samples, steps=128):
        device = next(self.parameters()).device
        N = vocab_size
        x = torch.randint(N, (n_samples, self.block_size), device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = ts[i].expand(n_samples)
            dt = ts[i] - ts[i + 1]
            _, sigma = noise(t)
            s = torch.exp(self(x, t).clamp(max=20))
            rate = sigma[:, None, None] / N * s
            rate.scatter_(-1, x[..., None], 0.0)
            probs = (rate * dt).clamp(0, 1)
            stay = (1 - probs.sum(-1, keepdim=True)).clamp_min(0)
            probs.scatter_(-1, x[..., None], stay)
            x = torch.multinomial(probs.view(-1, N), 1).view(n_samples, self.block_size)
        return x
