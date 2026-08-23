import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TextDiffusion, block_size, mask_id, topk_sample


@torch.no_grad()
def collect(seeds: int, steps: int, batch: int, ckpt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextDiffusion().to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    idx_rows, t_rows = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        x = torch.full((batch, block_size), mask_id, device=device, dtype=torch.long)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t = ts[i].expand(batch)
            for b in range(batch):
                idx_rows.append(x[b].cpu().numpy())
                t_rows.append(t[b].item())

            x0_hat = topk_sample(model(x, t))
            is_mask = x == mask_id
            unmask_p = (ts[i] - ts[i + 1]) / ts[i].clamp_min(1e-6)
            do = is_mask & (torch.rand(x.shape, device=device) < unmask_p)
            x = torch.where(do, x0_hat, x)

        is_mask = x == mask_id
        if is_mask.any():
            t = ts[-1].expand(batch)
            for b in range(batch):
                if is_mask[b].any():
                    idx_rows.append(x[b].cpu().numpy())
                    t_rows.append(t[b].item())
            x0_hat = topk_sample(model(x, t))
            x = torch.where(is_mask, x0_hat, x)

    idx = np.stack(idx_rows).astype(np.int64)
    t = np.array(t_rows, dtype=np.float32)
    return idx, t


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/ton-v1-mdlm-best.pt")
    p.add_argument("--out", default="artifacts/calib.npz")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    idx, t = collect(args.seeds, args.steps, args.batch, args.ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, idx=idx, t=t)
    print(f"wrote {args.out} | {len(t)} samples | idx {idx.shape} t {t.shape}")
