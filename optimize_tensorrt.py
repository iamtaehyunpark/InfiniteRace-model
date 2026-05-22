"""
Post-training inference optimisation — TensorRT or torch.compile fallback.

Only needed if run_training.sh Stage 6 reports inference > 50 ms.

Tries torch_tensorrt first; falls back to torch.compile with
mode="reduce-overhead" if torch_tensorrt is not installed.

Usage:
    python optimize_tensorrt.py \\
        --checkpoint checkpoints/lcm_final.pt \\
        --output     checkpoints/model_trt.pt \\
        --steps      2
"""
import argparse
import time
import sys

import torch


TARGET_MS = 50.0
WARMUP_RUNS = 10
BENCH_RUNS = 50


def _make_dummy_inputs(device: torch.device, dtype: torch.dtype):
    return (
        torch.randn(1, 3, 256, 256, device=device, dtype=dtype),
        torch.randn(1, 3, 256, 256, device=device, dtype=dtype),
        torch.randn(1, 256, 256, 2, device=device, dtype=dtype),
        torch.randn(1, 3, device=device, dtype=dtype),
    )


def _benchmark(model, inputs, steps: int, label: str) -> float:
    """Return mean inference time in ms over BENCH_RUNS iterations."""
    cue1, cue2, pos_map, action = inputs
    for _ in range(WARMUP_RUNS):
        model.infer(cue1, cue2, pos_map, action, steps=steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        model.infer(cue1, cue2, pos_map, action, steps=steps)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / BENCH_RUNS * 1000.0
    status = "OK" if ms <= TARGET_MS else "SLOW"
    print(f"  {label}: {ms:.1f} ms  ({1000.0 / ms:.1f} fps)  [{status}]")
    return ms


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optimise InfiniteRace world model with TensorRT or torch.compile."
    )
    p.add_argument(
        "--checkpoint", type=str, default="checkpoints/lcm_final.pt",
        help="Path to the LCM checkpoint to optimise",
    )
    p.add_argument(
        "--output", type=str, default="checkpoints/model_trt.pt",
        help="Path to save the optimised checkpoint",
    )
    p.add_argument(
        "--steps", type=int, default=2,
        help="Number of LCM inference steps to optimise for (1 or 2)",
    )
    return p


def main() -> None:
    args = _make_parser().parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16

    from world_model.model import InfiniteRaceWorldModel

    print(f"Loading checkpoint: {args.checkpoint}")
    model = InfiniteRaceWorldModel.load(args.checkpoint, device=str(device))
    model = model.half().eval()

    inputs = _make_dummy_inputs(device, dtype)

    print(f"\nBaseline ({args.steps}-step inference):")
    baseline_ms = _benchmark(model, inputs, args.steps, "baseline")

    # --- Attempt 1: torch_tensorrt ---
    trt_success = False
    try:
        import torch_tensorrt  # type: ignore[import]
        print("\nOptimising with torch_tensorrt ...")

        cue1, cue2, pos_map, action = inputs
        # Compile only the UNet (the hot path) to avoid TRT limitations on dynamic shapes
        model.unet = torch.compile(
            model.unet,
            backend="tensorrt",
            options={"truncate_long_and_double": True},
        )

        print("  torch_tensorrt compile complete.")
        print(f"\nBenchmark (torch_tensorrt, {args.steps}-step):")
        trt_ms = _benchmark(model, inputs, args.steps, "torch_tensorrt")
        trt_success = True
        opt_ms = trt_ms
        method = "torch_tensorrt"

    except (ImportError, Exception) as exc:
        print(f"  torch_tensorrt not available or failed: {exc}")
        print("  Falling back to torch.compile ...")

    # --- Attempt 2: torch.compile fallback ---
    if not trt_success:
        print("\nOptimising with torch.compile(mode='reduce-overhead') ...")
        model.unet = torch.compile(
            model.unet,
            mode="reduce-overhead",
            fullgraph=False,
        )

        # Warm up compilation (first pass triggers the compile)
        print("  Running warmup to trigger compilation ...")
        cue1, cue2, pos_map, action = inputs
        for _ in range(5):
            model.infer(cue1, cue2, pos_map, action, steps=args.steps)

        print(f"\nBenchmark (torch.compile, {args.steps}-step):")
        opt_ms = _benchmark(model, inputs, args.steps, "torch.compile")
        method = "torch.compile"

    # --- Results ---
    speedup = baseline_ms / opt_ms if opt_ms > 0 else float("inf")
    print(f"\nResults ({args.steps}-step):")
    print(f"  Baseline:    {baseline_ms:.1f} ms")
    print(f"  {method}: {opt_ms:.1f} ms")
    print(f"  Speedup:     {speedup:.2f}x")
    if opt_ms <= TARGET_MS:
        print(f"  Target ({TARGET_MS} ms): ACHIEVED")
    else:
        print(f"  Target ({TARGET_MS} ms): NOT achieved — consider running at 1-step inference")

    # --- Save ---
    # Save the model weights (torch.compile wraps in-place; VAE excluded as always)
    model.save(args.output)
    print(f"\nOptimised weights saved to: {args.output}")
    print("(Note: torch.compile graphs are session-local; load the .pt file")
    print(" and re-apply torch.compile at startup for best performance.)")


if __name__ == "__main__":
    main()
