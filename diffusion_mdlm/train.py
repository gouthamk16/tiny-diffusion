import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import TextDiffusion, decode, enc, mask_id, vocab_size

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
CACHE = str(ROOT / 'tinystories_gpt2_full.bin')
DEFAULT_CONFIG = ROOT / 'configs' / 'train_mdlm.yaml'


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
    mask = torch.rand(x0.shape, device=x0.device) < t[:, None]
    return torch.where(mask, mask_id, x0), mask


def dwdse_loss(model, x0, t):
    xt, mask = corrupt(x0, t)
    logits = model(xt, t)
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), x0.reshape(-1),
                         reduction='none').view(x0.shape)
    per_sample = (ce * mask).sum(-1) / t.clamp_min(1e-3)
    return per_sample.mean()


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
    n_embed, n_layers, n_heads, drop_rate = arch['n_embed'], arch['n_layers'], arch['n_heads'], arch['drop_rate']
    CKPT, BEST = ckpt_cfg['latest'], ckpt_cfg['best']

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

    gpt2 = None

    @torch.no_grad()
    def gen_quality(n_samples=8, steps=None):
        nonlocal gpt2
        if steps is None:
            steps = gen_steps
        if gpt2 is None:
            import importlib.util
            _orig = importlib.util.find_spec
            importlib.util.find_spec = lambda n, *a, **k: None if str(n).split('.')[0] == 'torchvision' else _orig(n, *a, **k)
            from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
            importlib.util.find_spec = _orig
            gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        model.eval()
        x = torch.cat([model.generate(n_samples=16, steps=steps)
                       for _ in range((n_samples + 15) // 16)], 0)[:n_samples]
        model.train()
        nll, ntok = 0.0, 0
        for i in range(0, n_samples, 8):
            ids = x[i:i + 8]
            logits = gpt2(ids).logits[:, :-1]
            tgt = ids[:, 1:]
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   reduction='sum').item()
            ntok += tgt.numel()
        ppl = math.exp(nll / ntok)
        bg, tot = set(), 0
        for r in x.tolist():
            for j in range(len(r) - 1):
                bg.add((r[j], r[j + 1]))
                tot += 1
        return ppl, len(bg) / max(1, tot), x

    def save_ckpt(path, epoch, best_val):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'epoch': epoch, 'lossi': lossi, 'best_val': best_val}, tmp)
        os.replace(tmp, path)

    start_epoch, best_val = 0, float('inf')
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start_epoch, lossi, best_val = ck['epoch'] + 1, ck['lossi'], ck['best_val']
        print(f"resumed from {CKPT} at step {start_epoch} (best val {best_val:.2f})")

    train_start = time.perf_counter()
    last_save = train_start
    timers = {'data': 0.0, 'loss': 0.0, 'backward': 0.0, 'step': 0.0}
    n_timed = 0

    for epoch in range(start_epoch, epochs):
        cur_lr = lr_at(epoch)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

        if epoch % eval_interval == 0:
            te = time.perf_counter()
            losses = estimate_loss()
            val = float(losses['val'])
            lossi.append(val)
            sync()
            best_tag = ""
            if val < best_val:
                best_val = val
                save_ckpt(BEST, epoch, best_val)
                best_tag = "  *BEST*"
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
            print(f"Step {epoch}/{epochs} : train {losses['train']:.4f} | val {val:.4f} "
                  f"| lr {cur_lr:.2e} | eval {eval_dt:.2f}s{mem}{best_tag}")

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

        if time.perf_counter() - last_save > 10:
            save_ckpt(CKPT, epoch, best_val)
            last_save = time.perf_counter()

    save_ckpt(CKPT, epochs - 1, best_val)
    losses = estimate_loss()
    val = float(losses['val'])
    lossi.append(val)
    if val < best_val:
        best_val = val
        save_ckpt(BEST, epochs - 1, best_val)
    stamp("training done", train_start, script_start)

    tg = time.perf_counter()
    ppl, distinct2, samples = gen_quality()
    stamp("quality eval", tg, script_start)
    print(f"FINAL @ step {epochs - 1} : val {val:.1f} | best_val {best_val:.1f} | "
          f"gen_ppl {ppl:.2f} | distinct2 {distinct2:.3f}")
    print("\n--- sample ---")
    print(decode(samples[0].tolist()))

    os.makedirs('artifacts', exist_ok=True)
    plt.plot([l.item() if torch.is_tensor(l) else l for l in lossi])
    plt.savefig('artifacts/loss.png')
    stamp("total", script_start, script_start)


if __name__ == '__main__':
    main()
