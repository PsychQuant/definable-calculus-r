#!/usr/bin/env python3
"""PyTorch sidecar for DD comprehensive benchmark.

Invoked from R via `system2()`. Reports median wall_ms over 5 reps as JSON.

Usage:
    python3 comprehensive-ad-comparison.py --expression sum_v2 --n 1000000 --backend torch_backward
"""
import argparse
import json
import statistics
import sys
import time


def make_f(name):
    import torch

    if name == "sum_v2":
        return lambda v: (v ** 2).sum()
    if name == "neg_sum_v2":
        return lambda v: -(v ** 2).sum()
    if name == "crossprod_v_v":
        return lambda v: torch.dot(v, v)
    if name == "sin_sum_v2":
        return lambda v: torch.sin((v ** 2).sum())
    if name == "sum_sin_v":
        return lambda v: torch.sin(v).sum()
    if name == "sum_exp_v":
        return lambda v: torch.exp(v).sum()
    # Phase A: additional vForce coverage
    if name == "sum_cos_v":
        return lambda v: torch.cos(v).sum()
    if name == "sum_log_v":
        return lambda v: torch.log(v).sum()
    if name == "sum_tanh_v":
        return lambda v: torch.tanh(v).sum()
    if name == "sum_sqrt_v":
        return lambda v: torch.sqrt(v).sum()
    if name == "scaled_sin":
        return lambda v: 2 * torch.sin(v).sum()
    if name == "neg_sum_cos":
        return lambda v: -torch.cos(v).sum()
    # Phase A: polynomial coverage
    if name == "sum_v3":
        return lambda v: (v ** 3).sum()
    if name == "sum_v4":
        return lambda v: (v ** 4).sum()
    if name == "sum_v5":
        return lambda v: (v ** 5).sum()
    # Phase A: Tier 2d composite coverage
    if name == "exp_sum_v2":
        return lambda v: torch.exp((v ** 2).sum())
    if name == "cos_crossprod":
        return lambda v: torch.cos(torch.dot(v, v))
    if name == "tanh_sum_sin":
        return lambda v: torch.tanh(torch.sin(v).sum())
    # Phase 1 (multi-axis benchmark expansion): Tier-3 walker fall-through cells
    if name == "sum_v_sin_v":
        return lambda v: (v * torch.sin(v)).sum()
    if name == "sum_sin_cos":
        return lambda v: (torch.sin(v) * torch.cos(v)).sum()
    if name == "sum_v_over_sin":
        return lambda v: (v / torch.sin(v)).sum()
    if name == "sum_sin_v_plus_1":
        return lambda v: torch.sin(v + 1).sum()
    if name == "sum_sin_2v":
        return lambda v: torch.sin(2 * v).sum()
    if name == "sum_v2_plus_sin":
        return lambda v: (v ** 2 + torch.sin(v)).sum()
    if name == "crossprod_v_sin":
        return lambda v: torch.dot(v, torch.sin(v))
    if name == "exp_log_sum_v2":
        return lambda v: torch.exp(torch.log((v ** 2).sum()))
    raise ValueError(f"unknown expression: {name}")


def run_torch_backward(f, n, reps=5):
    import torch

    torch.set_num_threads(1)
    times = []
    for _ in range(reps):
        v = torch.rand(n, dtype=torch.float64, requires_grad=True)
        t0 = time.perf_counter()
        loss = f(v)
        loss.backward()
        _ = v.grad
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def run_torch_func_grad(name, n, reps=5):
    import torch
    from torch.func import grad

    torch.set_num_threads(1)
    f = make_f(name)
    grad_f = grad(f)
    # Warm up to amortize first-call tracing
    v_warm = torch.rand(n, dtype=torch.float64)
    _ = grad_f(v_warm)
    times = []
    for _ in range(reps):
        v = torch.rand(n, dtype=torch.float64)
        t0 = time.perf_counter()
        _ = grad_f(v)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expression", required=True,
                   choices=["sum_v2", "neg_sum_v2", "crossprod_v_v",
                            "sin_sum_v2", "sum_sin_v", "sum_exp_v",
                            "sum_cos_v", "sum_log_v", "sum_tanh_v",
                            "sum_sqrt_v", "scaled_sin", "neg_sum_cos",
                            "sum_v3", "sum_v4", "sum_v5",
                            "exp_sum_v2", "cos_crossprod", "tanh_sum_sin",
                            # Phase 1: Tier-3 walker fall-through cells
                            "sum_v_sin_v", "sum_sin_cos", "sum_v_over_sin",
                            "sum_sin_v_plus_1", "sum_sin_2v",
                            "sum_v2_plus_sin", "crossprod_v_sin",
                            "exp_log_sum_v2"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--backend", required=True,
                   choices=["torch_backward", "torch_func_grad"])
    args = p.parse_args()

    try:
        if args.backend == "torch_backward":
            f = make_f(args.expression)
            wall = run_torch_backward(f, args.n)
        else:
            wall = run_torch_func_grad(args.expression, args.n)
        print(json.dumps({"wall_ms": wall}))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
