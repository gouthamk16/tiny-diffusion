import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from model import TextDiffusion, decode, enc, noise, vocab_size

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
CACHE = str(ROOT / 'tinystories_gpt2_full.bin')
DEFAULT_CONFIG = ROOT / 'configs' / 'train_sedd.yaml'


def stamp(msg, t0, script_start):
    dt = time.perf_counter() - t0
    print(f"[{time.perf_counter() - script_start:8.2f}s] {msg}: {dt:.3f}s")
    return time.perf_counter()


def build_cache(script_start):
    from datasets import load_dataset
    t0 = time.perf_counter()
    ds = load_dataset('karpathy/tinystories-gpt4-clean', split='train', streaming=True)
    eot = enc.eot_token
    tmp, total, report, batch = CACHE + '.tmp', 0, 50_000_000, []
    with open(tmp, 'wb') as f:
        def flush(texts):
            nonlocal total
            for ids in enc.encode_ordinary_batch(texts):
                ids.append(eot)
                np.array(ids, dtype=np.uint16).tofile(f)
                total += len(ids)
        for ex in ds:
            batch.append(ex['text'])
            if len(batch) == 1024:
                flush(batch)
                batch = []
                if total >= report:
                    stamp(f"  tokenized {total:,} tokens...", t0, script_start)
                    report += 50_000_000
        if batch:
            flush(batch)
    os.replace(tmp, CACHE)
    stamp(f"tokenized {total:,} tokens -> {CACHE}", t0, script_start)
    return np.fromfile(CACHE, dtype=np.uint16)


def corrupt(x0, t):
    lam, _ = noise(t)
    beta = torch.exp(-lam)
    replace = torch.rand(x0.shape, device=x0.device) < (1 - beta)[:, None]
    rand_tok = torch.randint(vocab_size, x0.shape, device=x0.device)
    return torch.where(replace, rand_tok, x0)


def dwdse_loss(model, x0, t):
    lam, dlam = noise(t)
    beta = torch.exp(-lam)
    a = beta + (1 - beta) / vocab_size
    b = (1 - beta) / vocab_size

    xt = corrupt(x0, t)
    log_s = model(xt, t).clamp(max=20)
    s = torch.exp(log_s)

    denom = torch.where(xt == x0, a[:, None], b[:, None])
    ra = a[:, None] / denom
    rb = b[:, None] / denom
    Ka = ra * (ra.clamp_min(1e-9).log() - 1)
    Kb = rb * (rb.clamp_min(1e-9).log() - 1)

    ls_x0 = log_s.gather(-1, x0[..., None]).squeeze(-1)
    ls_xt = log_s.gather(-1, xt[..., None]).squeeze(-1)
    s_xt = s.gather(-1, xt[..., None]).squeeze(-1)

    full = s.sum(-1) - (rb * log_s.sum(-1) + (ra - rb) * ls_x0) + ((vocab_size - 1) * Kb + Ka)
    is_xt_x0 = xt == x0
    term_xt = s_xt - torch.where(is_xt_x0, ra, rb) * ls_xt + torch.where(is_xt_x0, Ka, Kb)
    return (dlam[:, None] * (full - term_xt)).mean()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8')) or {}
    hp, arch, ckpt_cfg = cfg['train'], cfg['arch'], cfg['ckpt']

    batch_size = hp['batch_size']
    block_size = hp['block_size']
    epochs = hp['epochs']
    lr = hp['lr']
    min_lr = hp['min_lr']
    warmup_steps = hp['warmup_steps']
    eval_interval = hp['eval_interval']
    eval_iters = hp['eval_iters']
    gen_steps = hp['gen_steps']
    checkpoint_interval = hp.get('checkpoint_interval', 1000)
    n_embed, n_layers, n_heads, drop_rate = arch['n_embed'], arch['n_layers'], arch['n_heads'], arch['drop_rate']
    CKPT = ckpt_cfg['latest']

    script_start = time.perf_counter()
    torch.manual_seed(cfg.get('seed', 1337))
    print(f"config: {args.config}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, epochs - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))

    t0 = time.perf_counter()
    if os.path.exists(CACHE):
        arr = np.fromfile(CACHE, dtype=np.uint16)
        stamp(f"loaded cache {len(arr):,} tokens", t0, script_start)
    else:
        arr = build_cache(script_start)

    n = int(0.9 * len(arr))
    train_data = arr[:n]
    val_data = arr[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x0 = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
        return x0.to(device)

    t0 = time.perf_counter()
    model = TextDiffusion(n_embed, n_layers, n_heads, block_size, drop_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    lossi = []
    stamp(f"model built ({n_params/1e6:.1f}M params)", t0, script_start)

    @torch.no_grad()
    def estimate_loss():
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if device == 'cuda' else None
        torch.manual_seed(1234)
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x0 = get_batch(split)
                t = torch.rand(x0.shape[0], device=device)
                losses[k] = dwdse_loss(model, x0, t).item()
            out[split] = losses.mean()
        model.train()
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        return out

    def save_ckpt(path, epoch):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'epoch': epoch, 'lossi': lossi}, tmp)
        os.replace(tmp, path)

    start_epoch = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start_epoch = ck['epoch'] + 1
        lossi = ck['lossi']
        print(f"resumed from {CKPT} at step {start_epoch}")

    train_start = time.perf_counter()
    timers = {'data': 0.0, 'loss': 0.0, 'backward': 0.0, 'step': 0.0}
    n_timed = 0

    for epoch in range(start_epoch, epochs):
        cur_lr = lr_at(epoch)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

        if epoch % eval_interval == 0:
            te = time.perf_counter()
            losses = estimate_loss()
            lossi.append(losses['val'])
            sync()
            eval_dt = time.perf_counter() - te
            if n_timed:
                per = {k: 1000 * v / n_timed for k, v in timers.items()}
                tot = sum(per.values())
                sps = 1000 / tot if tot else 0
                print(f"  timing/step: data {per['data']:.1f}ms | loss {per['loss']:.1f}ms | "
                      f"backward {per['backward']:.1f}ms | opt {per['step']:.1f}ms | "
                      f"{sps:.1f} steps/s | {sps*batch_size*block_size:,.0f} tok/s")
                timers = {k: 0.0 for k in timers}
                n_timed = 0
            mem = f" | gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == 'cuda' else ""
            print(f"Step {epoch}/{epochs} : train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {cur_lr:.2e} | eval {eval_dt:.2f}s{mem}")

        sync()
        t_a = time.perf_counter()
        x0 = get_batch('train')
        B = x0.shape[0]
        t = (torch.arange(B, device=device) + torch.rand(B, device=device)) / B
        sync()
        t_b = time.perf_counter()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = dwdse_loss(model, x0, t)
        sync()
        t_c = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sync()
        t_d = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync()
        t_e = time.perf_counter()

        timers['data'] += t_b - t_a
        timers['loss'] += t_c - t_b
        timers['backward'] += t_d - t_c
        timers['step'] += t_e - t_d
        n_timed += 1

        if epoch % checkpoint_interval == 0 and epoch > start_epoch:
            save_ckpt(CKPT, epoch)

    save_ckpt(CKPT, epochs - 1)
    stamp("training done", train_start, script_start)

    losses = estimate_loss()
    lossi.append(losses['val'])
    mem = f" | gpu {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == 'cuda' else ""
    print(f"Step {epoch}/{epochs} : train {losses['train']:.4f} | val {losses['val']:.4f} "
          f"| lr {lr_at(epoch):.2e} | eval --{mem}")

    tg = time.perf_counter()
    sample = model.generate(n_samples=1, steps=gen_steps)
    stamp(f"generation ({gen_steps} steps)", tg, script_start)
    print("\n--- sample ---")
    print(decode(sample[0].tolist()))

    os.makedirs('artifacts', exist_ok=True)
    plt.plot([l.item() if torch.is_tensor(l) else l for l in lossi])
    plt.savefig('artifacts/loss.png')
    stamp("total", script_start, script_start)


if __name__ == '__main__':
    main()
