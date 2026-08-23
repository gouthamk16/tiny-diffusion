import argparse
import sys
import time
from pathlib import Path

import torch
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TextDiffusion, vocab_size


def load_engine(path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        return trt.Runtime(logger).deserialize_cuda_engine(f.read())


def run_trt(engine, batch: int, warmup: int, iters: int):
    ctx = engine.create_execution_context()
    idx = torch.randint(0, vocab_size + 1, (batch, 128), device="cuda", dtype=torch.int64)
    t = torch.rand(batch, device="cuda", dtype=torch.float32)

    ctx.set_input_shape("idx", tuple(idx.shape))
    ctx.set_input_shape("t", tuple(t.shape))
    logits = torch.empty(tuple(ctx.get_tensor_shape("logits")), device="cuda", dtype=torch.float32)

    ctx.set_tensor_address("idx", idx.data_ptr())
    ctx.set_tensor_address("t", t.data_ptr())
    ctx.set_tensor_address("logits", logits.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return logits, ms


@torch.no_grad()
def run_pytorch(batch: int, warmup: int, iters: int, ckpt: str):
    model = TextDiffusion().cuda().eval()
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])

    idx = torch.randint(0, vocab_size + 1, (batch, 128), device="cuda", dtype=torch.long)
    t = torch.rand(batch, device="cuda")

    for _ in range(warmup):
        model(idx, t)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        model(idx, t)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return model(idx, t), ms


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="artifacts/mdlm_best_fp16.engine")
    p.add_argument("--ckpt", default="artifacts/ckpt_best.pt")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--compare-pytorch", action="store_true")
    args = p.parse_args()

    engine = load_engine(args.engine)
    logits, trt_ms = run_trt(engine, args.batch, args.warmup, args.iters)
    print(f"trt logits {tuple(logits.shape)} | {trt_ms:.2f} ms/step (batch={args.batch})")

    if args.compare_pytorch:
        pt_logits, pt_ms = run_pytorch(args.batch, args.warmup, args.iters, args.ckpt)
        diff = (logits - pt_logits).abs().max().item()
        print(f"pt  logits {tuple(pt_logits.shape)} | {pt_ms:.2f} ms/step")
        print(f"max |trt - pt| = {diff:.4f}")
