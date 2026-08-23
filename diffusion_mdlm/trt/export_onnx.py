import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TextDiffusion, vocab_size


def export(ckpt: str, out: str):
    device = "cpu"
    model = TextDiffusion().to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    def _rms_norm_onnx(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * self.weight

    for m in model.modules():
        if isinstance(m, nn.RMSNorm):
            m.forward = types.MethodType(_rms_norm_onnx, m)

    B, T = 1, 128
    idx = torch.randint(0, vocab_size + 1, (B, T), dtype=torch.long, device=device)
    t = torch.rand(B, device=device)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (idx, t),
        out,
        input_names=["idx", "t"],
        output_names=["logits"],
        dynamic_axes={
            "idx": {0: "batch", 1: "seq"},
            "t": {0: "batch"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/ton-v1-mdlm-best.pt")
    p.add_argument("--out", default="artifacts/mdlm_best.onnx")
    args = p.parse_args()
    export(args.ckpt, args.out)
