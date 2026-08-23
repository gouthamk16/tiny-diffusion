import argparse
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import modelopt.onnx.autocast as autocast
from modelopt.onnx.quantization import quantize

SEQ = 128
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 1, 8
CALIB_SHAPES = f"idx:{OPT_BATCH}x{SEQ},t:{OPT_BATCH}"


def prepare_onnx(onnx_path: Path, precision: str, calib: Path | None, max_samples: int) -> Path:
    if precision == "fp32":
        return onnx_path

    cast_path = onnx_path.with_name(onnx_path.stem + f"_{precision}.onnx")
    if precision in ("fp16", "bf16"):
        model = autocast.convert_to_mixed_precision(
            onnx_path=str(onnx_path),
            low_precision_type=precision,
            keep_io_types=True,
        )
        onnx.save(model, str(cast_path))
    elif precision == "fp8":
        calib_data = None
        if calib:
            data = np.load(calib)
            calib_data = {k: data[k] for k in data.files}
            n = calib_data["t"].shape[0]
            if n > max_samples:
                pick = np.random.default_rng(0).choice(n, max_samples, replace=False)
                calib_data = {k: v[pick] for k, v in calib_data.items()}
                print(f"calib {calib} | subsampled {max_samples}/{n}")
            else:
                print(f"calib {calib} | {n} samples")
        quantize(
            onnx_path=str(onnx_path),
            quantize_mode="fp8",
            output_path=str(cast_path),
            calibration_shapes=CALIB_SHAPES,
            calibration_data=calib_data,
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")

    print(f"cast {cast_path}")
    return cast_path


def build(onnx_path: Path, engine_path: Path, precision: str, calib: Path | None, max_samples: int):
    onnx_path = prepare_onnx(onnx_path, precision, calib, max_samples)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape("idx", (MIN_BATCH, SEQ), (OPT_BATCH, SEQ), (MAX_BATCH, SEQ))
    profile.set_shape("t", (MIN_BATCH,), (OPT_BATCH,), (MAX_BATCH,))
    config.add_optimization_profile(profile)

    print(f"building {engine_path} ({precision})...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine))
    print(f"wrote {engine_path} ({engine.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="artifacts/mdlm_best.onnx")
    p.add_argument("--engine", default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"], default="fp16")
    p.add_argument("--calib", default=None, help="calib.npz from collect_calib.py (fp8 only)")
    p.add_argument("--max-samples", type=int, default=512, help="fp8 calib cap (speed vs accuracy)")
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    suffix = "" if args.precision == "fp32" else f"_{args.precision}"
    engine_path = Path(args.engine or onnx_path.with_name(onnx_path.stem + suffix + ".engine"))
    calib = Path(args.calib) if args.calib else None
    build(onnx_path, engine_path, args.precision, calib, args.max_samples)
