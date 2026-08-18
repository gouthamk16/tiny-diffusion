import argparse
import json
import statistics as st
import time

import torch
import tensorrt as trt
from gen_mdlm import block_size, decode, mask_id, topk_sample

def safe_topk_sample(logits, k=20):
    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    return topk_sample(logits, k)

def load_engine(path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        return trt.Runtime(logger).deserialize_cuda_engine(f.read())

class TrtModel:
    def __init__(self, engine):
        self.ctx = engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, idx: torch.Tensor, t: torch.Tensor):
        self.ctx.set_input_shape("idx", tuple(idx.shape))
        self.ctx.set_input_shape("t", tuple(t.shape))
        logits = torch.empty(tuple(self.ctx.get_tensor_shape("logits")), device="cuda", dtype=torch.float32)
        self.ctx.set_tensor_address("idx", idx.data_ptr())
        self.ctx.set_tensor_address("t", t.data_ptr())
        self.ctx.set_tensor_address("logits", logits.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return logits

def sync():
    torch.cuda.synchronize()

def denoise_step(model, x, ts, i, n, k=20):
    t = ts[i].expand(n)
    x0_hat = safe_topk_sample(model(x, t), k)
    is_mask = x == mask_id
    unmask_p = (ts[i] - ts[i + 1]) / ts[i].clamp_min(1e-6)
    do = is_mask & (torch.rand(x.shape, device="cuda") < unmask_p)
    return torch.where(do, x0_hat, x)

@torch.no_grad()
def generate(model, n_samples, steps, k=20):
    x = torch.full((n_samples, block_size), mask_id, device="cuda", dtype=torch.long)
    ts = torch.linspace(1.0, 0.0, steps + 1, device="cuda")
    for i in range(steps):
        x = denoise_step(model, x, ts, i, n_samples, k)
    is_mask = x == mask_id
    if is_mask.any():
        x0_hat = safe_topk_sample(model(x, ts[-1].expand(n_samples)), k)
        x = torch.where(is_mask, x0_hat, x)
    return x

def timed_generate(model, n_samples, steps, track_mem=True):
    if track_mem:
        torch.cuda.reset_peak_memory_stats()
    x = torch.full((n_samples, block_size), mask_id, device="cuda", dtype=torch.long)
    ts = torch.linspace(1.0, 0.0, steps + 1, device="cuda")

    sync(); a = time.perf_counter()
    x = denoise_step(model, x, ts, 0, n_samples)
    sync(); prefill = (time.perf_counter() - a) * 1000

    sync(); a = time.perf_counter()
    for i in range(1, steps):
        x = denoise_step(model, x, ts, i, n_samples)
    is_mask = x == mask_id
    if is_mask.any():
        x0_hat = safe_topk_sample(model(x, ts[-1].expand(n_samples)))
        x = torch.where(is_mask, x0_hat, x)
    sync(); decode = (time.perf_counter() - a) * 1000

    peak_a = torch.cuda.max_memory_allocated() / 1e6
    peak_r = torch.cuda.max_memory_reserved() / 1e6
    return prefill, decode, peak_a, peak_r

def bench(engine_path: str, n_runs: int, n_samples: int, steps: int):
    engine = load_engine(engine_path)
    model = TrtModel(engine)
    engine_mb = len(open(engine_path, "rb").read()) / 1e6

    cold_pf, cold_dc, _, _ = timed_generate(model, n_samples, steps, track_mem=False)
    cold_total = cold_pf + cold_dc
    for _ in range(2):
        timed_generate(model, n_samples, steps, track_mem=False)

    runs = []
    for _ in range(n_runs):
        pf, dc, pa, pr = timed_generate(model, n_samples, steps)
        runs.append({
            "prefill_ms": pf, "decode_ms": dc, "total_ms": pf + dc,
            "decode_per_step_ms": dc / (steps - 1),
            "peak_mem_alloc_mb": pa, "peak_mem_reserved_mb": pr,
        })

    def summ(key):
        v = [r[key] for r in runs]
        return {"mean": st.mean(v), "std": st.pstdev(v), "min": min(v), "max": max(v)}

    return {
        "config": {
            "engine": engine_path, "engine_size_mb": round(engine_mb, 1),
            "n_samples": n_samples, "denoising_steps": steps, "block_size": block_size,
            "device": "cuda", "gpu": torch.cuda.get_device_name(0),
        },
        "note": ("TensorRT MDLM generation benchmark. 'prefill' = first denoising step, "
                 "'decode' = remaining steps. Sampling/top-k runs on GPU in PyTorch."),
        "cold_start_total_ms": cold_total,
        "n_runs": n_runs,
        "summary": {k: summ(k) for k in
                    ["prefill_ms", "decode_ms", "decode_per_step_ms", "total_ms",
                     "peak_mem_alloc_mb", "peak_mem_reserved_mb"]},
        "runs": runs,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="mdlm_best_fp16.engine")
    p.add_argument("--out", default=None)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--print", dest="print_samples", action="store_true")
    args = p.parse_args()

    if args.print_samples:
        engine = load_engine(args.engine)
        model = TrtModel(engine)
        torch.manual_seed(0)
        out = generate(model, args.n_samples, args.steps, args.topk)
        print(f"engine: {args.engine} | {args.steps} steps | n={args.n_samples}")
        for j, row in enumerate(out.tolist()):
            print(f"\n--- sample {j+1} ---")
            print(decode(row))
        raise SystemExit(0)

    out_path = args.out or args.engine.replace(".engine", "_timing.json")
    out = bench(args.engine, args.n_runs, args.n_samples, args.steps)
    json.dump(out, open(out_path, "w"), indent=2)

    s = out["summary"]
    cfg = out["config"]
    print(f"engine: {cfg['engine']} ({cfg['engine_size_mb']:.0f} MB) | gpu: {cfg['gpu']}")
    print(f"cold-start total: {out['cold_start_total_ms']:.1f} ms")
    print(f"prefill (1 step):   {s['prefill_ms']['mean']:.2f} +/- {s['prefill_ms']['std']:.2f} ms")
    print(f"decode ({args.steps - 1} steps): {s['decode_ms']['mean']:.1f} +/- {s['decode_ms']['std']:.1f} ms "
          f"({s['decode_per_step_ms']['mean']:.2f} ms/step)")
    print(f"total  ({args.steps} steps): {s['total_ms']['mean']:.1f} +/- {s['total_ms']['std']:.1f} ms")
    print(f"peak mem alloc: {s['peak_mem_alloc_mb']['mean']:.0f} MB | reserved: {s['peak_mem_reserved_mb']['mean']:.0f} MB")
    print(f"saved {out_path}")
