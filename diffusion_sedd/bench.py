import argparse
import json
import os
import statistics as st
import time

import torch

from model import TextDiffusion, block_size, noise, vocab_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sync():
    if device == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def denoise_step(model, x, ts, i, n):
    t = ts[i].expand(n)
    dt = ts[i] - ts[i + 1]
    _, sigma = noise(t)
    s = torch.exp(model(x, t).clamp(max=20))
    rate = sigma[:, None, None] / vocab_size * s
    rate.scatter_(-1, x[..., None], 0.0)
    probs = (rate * dt).clamp(0, 1)
    stay = (1 - probs.sum(-1, keepdim=True)).clamp_min(0)
    probs.scatter_(-1, x[..., None], stay)
    return torch.multinomial(probs.view(-1, vocab_size), 1).view(n, x.shape[1])


@torch.no_grad()
def timed_generate(model, n_samples, steps, track_mem=True):
    if track_mem and device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    x = torch.randint(vocab_size, (n_samples, model.block_size), device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    sync()
    a = time.perf_counter()
    x = denoise_step(model, x, ts, 0, n_samples)
    sync()
    prefill = (time.perf_counter() - a) * 1000
    sync()
    a = time.perf_counter()
    for i in range(1, steps):
        x = denoise_step(model, x, ts, i, n_samples)
    sync()
    decode = (time.perf_counter() - a) * 1000
    peak_a = torch.cuda.max_memory_allocated() / 1e6 if device == 'cuda' else 0.0
    peak_r = torch.cuda.max_memory_reserved() / 1e6 if device == 'cuda' else 0.0
    return prefill, decode, peak_a, peak_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--n-samples', dest='n_samples', type=int, default=None)
    p.add_argument('--n-runs', dest='n_runs', type=int, default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding='utf-8')) or {}
    n_runs = args.n_runs or cfg.get('n_runs') or 20
    n_samples = args.n_samples or cfg.get('n_samples') or 1
    steps = args.steps or cfg.get('steps') or 128
    ckpt = args.ckpt or cfg.get('ckpt') or 'artifacts/ckpt.pt'
    out_path = args.out or 'artifacts/gen_timing.json'

    arch = cfg.get('arch') or {}
    torch.manual_seed(0)
    model = TextDiffusion(
        n_embed=arch.get('n_embed', 512),
        n_layers=arch.get('n_layers', 6),
        n_heads=arch.get('n_heads', 16),
        block_size=arch.get('block_size', 128),
        drop_rate=arch.get('drop_rate', 0.0),
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    cold_pf, cold_dc, _, _ = timed_generate(model, n_samples, steps, track_mem=False)
    cold_total = cold_pf + cold_dc
    for _ in range(2):
        timed_generate(model, n_samples, steps, track_mem=False)

    runs = []
    for _ in range(n_runs):
        pf, dc, pa, pr = timed_generate(model, n_samples, steps)
        runs.append({"prefill_ms": pf, "decode_ms": dc, "total_ms": pf + dc,
                     "decode_per_step_ms": dc / (steps - 1),
                     "peak_mem_alloc_mb": pa, "peak_mem_reserved_mb": pr})

    def summ(key):
        v = [r[key] for r in runs]
        return {"mean": st.mean(v), "std": st.pstdev(v), "min": min(v), "max": max(v)}

    out = {
        "config": {"n_samples": n_samples, "denoising_steps": steps, "block_size": block_size,
                   "device": device, "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else "cpu",
                   "checkpoint_step": int(ck['epoch']), "model_param_mem_mb": round(param_mem, 1)},
        "note": ("Non-autoregressive diffusion: no prompt/KV-cache. Generation is `denoising_steps` "
                 "full-sequence bidirectional forward passes. 'prefill' = first denoising step, "
                 "'decode' = remaining steps (each the same cost as prefill here)."),
        "cold_start_total_ms": cold_total,
        "n_runs": n_runs,
        "summary": {k: summ(k) for k in
                    ["prefill_ms", "decode_ms", "decode_per_step_ms", "total_ms",
                     "peak_mem_alloc_mb", "peak_mem_reserved_mb"]},
        "runs": runs,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=2)

    s = out["summary"]
    print(f"gpu: {out['config']['gpu']} | model params: {param_mem:.0f} MB")
    print(f"cold-start total: {cold_total:.1f} ms")
    print(f"prefill (1 step):   {s['prefill_ms']['mean']:.2f} +/- {s['prefill_ms']['std']:.2f} ms")
    print(f"decode ({steps-1} steps): {s['decode_ms']['mean']:.1f} +/- {s['decode_ms']['std']:.1f} ms "
          f"({s['decode_per_step_ms']['mean']:.2f} ms/step)")
    print(f"total  ({steps} steps): {s['total_ms']['mean']:.1f} +/- {s['total_ms']['std']:.1f} ms")
    print(f"peak mem alloc: {s['peak_mem_alloc_mb']['mean']:.0f} MB | reserved: {s['peak_mem_reserved_mb']['mean']:.0f} MB")
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
