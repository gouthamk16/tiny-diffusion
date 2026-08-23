import argparse
import sys

import torch

from model import TextDiffusion, decode

sys.stdout.reconfigure(encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--n-samples', dest='n_samples', type=int, default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding='utf-8')) or {}
    ckpt = args.ckpt or cfg.get('ckpt') or 'artifacts/ton-v1-sedd-latest.pt'
    steps = args.steps or cfg.get('steps') or 256
    n_samples = args.n_samples or cfg.get('n_samples') or 6
    device = args.device or cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

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
    print(f"loaded {ckpt} @ step {ck['epoch']}")
    model.eval()

    print(f"\n{'='*70}\n{steps} denoising steps\n{'='*70}")
    out = model.generate(n_samples=n_samples, steps=steps)
    for j, row in enumerate(out.tolist()):
        print(f"\n--- sample {j+1} ({steps} steps) ---")
        print(decode(row))


if __name__ == '__main__':
    main()
