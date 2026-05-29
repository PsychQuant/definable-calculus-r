#!/usr/bin/env python3
"""JAX sidecar for DD comprehensive benchmark — Phase 3.

Invoked from R via system2(). Reports median wall_ms over 5 reps as JSON.

Two backends:
  jax_grad  — jax.grad without JIT (symbolic-derivative + per-call trace)
  jax_jit   — jax.jit(jax.grad(...)) with 3 warm-up calls before timing

Usage:
    python3 sidecar_jax_compare.py --expression sum_v2 --n 1000000 --backend jax_grad
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def make_jax_fn(name: str):
    import jax.numpy as jnp

    table = {
        "sum_v2":     lambda v: jnp.sum(v ** 2),
        "sum_sin_v":  lambda v: jnp.sum(jnp.sin(v)),
        "sin_sum_v2": lambda v: jnp.sin(jnp.sum(v ** 2)),
        "sum_v3":     lambda v: jnp.sum(v ** 3),
    }
    if name not in table:
        raise ValueError(f"unknown expression: {name}")
    return table[name]


def run_jax_grad(name: str, n: int, reps: int = 5) -> float:
    import jax
    import jax.numpy as jnp
    import numpy as np

    grad_f = jax.grad(make_jax_fn(name))
    times: list[float] = []
    for _ in range(reps):
        v = jnp.asarray(np.random.rand(n), dtype=jnp.float64)
        t0 = time.perf_counter()
        out = grad_f(v)
        out.block_until_ready()  # JAX is async; force completion before timing
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def run_jax_jit(name: str, n: int, reps: int = 5, warmup: int = 3) -> float:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jit_grad_f = jax.jit(jax.grad(make_jax_fn(name)))
    # Warm-up: amortize JIT trace + XLA compile (multi-second on first call)
    v_warm = jnp.asarray(np.random.rand(n), dtype=jnp.float64)
    for _ in range(warmup):
        out = jit_grad_f(v_warm)
        out.block_until_ready()
    times: list[float] = []
    for _ in range(reps):
        v = jnp.asarray(np.random.rand(n), dtype=jnp.float64)
        t0 = time.perf_counter()
        out = jit_grad_f(v)
        out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expression", required=True,
                   choices=["sum_v2", "sum_sin_v", "sin_sum_v2", "sum_v3"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--backend", required=True,
                   choices=["jax_grad", "jax_jit"])
    args = p.parse_args()

    try:
        # JAX defaults to float32 unless x64 enabled — match DD's float64 precision.
        from jax import config as jax_config
        jax_config.update("jax_enable_x64", True)

        if args.backend == "jax_grad":
            wall = run_jax_grad(args.expression, args.n)
        else:
            wall = run_jax_jit(args.expression, args.n)
        print(json.dumps({"wall_ms": wall}))
        sys.exit(0)
    except ImportError as e:
        print(json.dumps({"error": f"jax not installed: {e}"}))
        sys.exit(2)  # distinct exit code for missing-dep
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
