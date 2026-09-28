#!/usr/bin/env python3
"""Train and freeze the T4 top-k router; print clean ACC and write stats."""

from __future__ import annotations

import argparse

from router import train_and_freeze


def main() -> None:
    p = argparse.ArgumentParser(description="T4: train learned top-k router")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()
    out = train_and_freeze(
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(
        f"RESULT param_count={out['param_count']} "
        f"MACs={out['macs_per_decision']} "
        f"clean_ACC={out['clean_acc']:.4f} "
        f"sha256={out['weights_sha256'][:16]}…"
    )


if __name__ == "__main__":
    main()
