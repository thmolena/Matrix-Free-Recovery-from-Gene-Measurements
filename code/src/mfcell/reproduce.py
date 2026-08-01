"""Console entry point for reproducing and verifying the manuscript evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment, verify_locked_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the real PBMC3k matrix-free recovery experiment"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify",
        action="store_true",
        help="recompute scientific outputs and compare with the locked artifact",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="recompute and replace source-tree results, tables, and figures",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        help="write freshly recomputed JSON after a successful --verify",
    )
    args = parser.parse_args()
    if args.verification_output is not None and not args.verify:
        parser.error("--verification-output requires --verify")
    results = (
        verify_locked_results()
        if args.verify
        else run_experiment(write_artifacts=args.write)
    )
    if args.verification_output is not None:
        args.verification_output.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "status": "verified" if args.verify else ("written" if args.write else "reproduced"),
        "primary_method_mse": results["primary"]["metrics"]["method"]["mse"],
        "primary_extra_trees_mse": results["primary"]["metrics"]["extra_trees"]["mse"],
        "external_method_mse": results["external"]["metrics"]["method"]["mse"],
        "external_extra_trees_mse": results["external"]["metrics"]["extra_trees"]["mse"],
        "all_cg_converged": (
            results["primary"]["solver"]["all_converged"]
            and results["external"]["solver"]["all_converged"]
        ),
        "data_sha256": {
            key: value["sha256"] for key, value in results["data"].items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
