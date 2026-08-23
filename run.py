#!/usr/bin/env python3
"""Train or infer SEDD / MDLM from a YAML config at the repo root."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = {"mdlm": ROOT / "diffusion_mdlm", "sedd": ROOT / "diffusion_sedd"}
TRAIN = {"mdlm": "train.py", "sedd": "train.py"}
GEN = {"mdlm": "gen.py", "sedd": "gen.py"}
BENCH = {"mdlm": "bench.py", "sedd": "bench.py"}


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def overlay(cfg, args):
    for key in ("backend", "precision", "ckpt", "steps", "n_samples", "engine"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if getattr(args, "rebuild", False):
        cfg["rebuild"] = True
    return cfg


def run(cmd, cwd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, cwd=cwd)


def cmd_train(cfg, config_path):
    model = cfg.get("model")
    if model not in DIRS:
        sys.exit(f"unknown model {model!r}; expected mdlm or sedd")
    run([sys.executable, TRAIN[model], "--config", str(config_path.resolve())], DIRS[model])


def cmd_infer(cfg, config_path, args):
    model = cfg.get("model")
    if model not in DIRS:
        sys.exit(f"unknown model {model!r}; expected mdlm or sedd")
    backend = cfg.get("backend", "torch")
    cwd = DIRS[model]

    if backend == "torch":
        cmd = [sys.executable, GEN[model], "--config", str(config_path.resolve())]
        if args.ckpt is not None:
            cmd += ["--ckpt", args.ckpt]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]
        if args.n_samples is not None:
            cmd += ["--n-samples", str(args.n_samples)]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd, cwd)
        return

    if backend != "tensorrt":
        sys.exit(f"unknown backend {backend!r}; expected torch or tensorrt")
    engine = trt_engine(cfg, args, cwd)
    steps = cfg.get("steps", 256)
    n_samples = cfg.get("n_samples", 6)
    topk = cfg.get("topk", 20)
    gen = [sys.executable, "trt/bench.py", "--engine", engine, "--print",
           "--n-samples", str(n_samples), "--steps", str(steps), "--topk", str(topk)]
    run(gen, cwd)


def trt_engine(cfg, args, cwd):
    if cfg.get("model") != "mdlm":
        sys.exit("TensorRT is only wired for MDLM")
    precision = cfg.get("precision", "fp16")
    ckpt = cfg.get("ckpt", "artifacts/ton-v1-mdlm-best.pt")
    onnx = cfg.get("onnx", "artifacts/mdlm_best.onnx")
    stem = Path(onnx).stem
    suffix = "" if precision == "fp32" else f"_{precision}"
    default_engine = str(Path(onnx).with_name(f"{stem}{suffix}.engine"))
    engine = args.engine or cfg.get("engine") or default_engine
    if args.precision:
        engine = args.engine or default_engine
    calib = cfg.get("calib", "artifacts/calib.npz")
    rebuild = bool(cfg.get("rebuild", False))
    py = sys.executable
    if rebuild or not (cwd / onnx).exists():
        run([py, "trt/export_onnx.py", "--ckpt", ckpt, "--out", onnx], cwd)
    if precision == "fp8" and (rebuild or not (cwd / calib).exists()):
        run([py, "trt/collect_calib.py", "--ckpt", ckpt, "--out", calib], cwd)
    if rebuild or not (cwd / engine).exists():
        build = [py, "trt/build.py", "--onnx", onnx, "--engine", engine, "--precision", precision]
        if precision == "fp8":
            build += ["--calib", calib]
        run(build, cwd)
    return engine


def cmd_bench(cfg, config_path, args):
    model = cfg.get("model")
    if model not in DIRS:
        sys.exit(f"unknown model {model!r}; expected mdlm or sedd")
    backend = cfg.get("backend", "torch")
    cwd = DIRS[model]
    n_runs = args.n_runs or 20
    n_samples = args.n_samples if args.n_samples is not None else 1
    steps = args.steps or cfg.get("steps")
    ckpt = args.ckpt or cfg.get("ckpt")

    if backend == "torch":
        cmd = [sys.executable, BENCH[model], "--config", str(config_path.resolve()),
               "--n-samples", str(n_samples), "--n-runs", str(n_runs)]
        if ckpt:
            cmd += ["--ckpt", ckpt]
        if steps:
            cmd += ["--steps", str(steps)]
        run(cmd, cwd)
        return

    if backend != "tensorrt":
        sys.exit(f"unknown backend {backend!r}; expected torch or tensorrt")
    engine = trt_engine(cfg, args, cwd)
    cmd = [sys.executable, "trt/bench.py", "--engine", engine,
           "--n-samples", str(n_samples), "--n-runs", str(n_runs)]
    if steps:
        cmd += ["--steps", str(steps)]
    run(cmd, cwd)


def main():
    p = argparse.ArgumentParser(description="TON-v1 train / infer from YAML")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--config", required=True)

    pi = sub.add_parser("infer")
    pi.add_argument("--config", required=True)
    pi.add_argument("--backend", choices=["torch", "tensorrt"])
    pi.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"])
    pi.add_argument("--ckpt")
    pi.add_argument("--steps", type=int)
    pi.add_argument("--n-samples", dest="n_samples", type=int)
    pi.add_argument("--engine")
    pi.add_argument("--device")
    pi.add_argument("--rebuild", action="store_true")

    pb = sub.add_parser("bench")
    pb.add_argument("--config", required=True)
    pb.add_argument("--backend", choices=["torch", "tensorrt"])
    pb.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"])
    pb.add_argument("--ckpt")
    pb.add_argument("--steps", type=int)
    pb.add_argument("--n-samples", dest="n_samples", type=int)
    pb.add_argument("--n-runs", dest="n_runs", type=int)
    pb.add_argument("--engine")
    pb.add_argument("--rebuild", action="store_true")

    args = p.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")
    cfg = load_yaml(config_path)
    if args.cmd == "train":
        cmd_train(cfg, config_path)
    elif args.cmd == "bench":
        cfg = overlay(cfg, args)
        cmd_bench(cfg, config_path, args)
    else:
        cfg = overlay(cfg, args)
        cmd_infer(cfg, config_path, args)


if __name__ == "__main__":
    main()
