"""Command-line interface for toptimization.

Usage:
  toptimization run <problem.yaml> [options]
  toptimization validate <problem.yaml>

Run options:
  --backend   metal|cpu|cuda|vulkan|auto  (default: auto)
  --no-viz    Disable live matplotlib visualization
  --no-live   Disable live update; only save final image
  --output    Override output directory
  --verbose   Extra logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="toptimization",
        description="GPU-accelerated topology optimization (SIMP method)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run topology optimization")
    run_parser.add_argument("problem", help="Path to YAML problem definition file")
    run_parser.add_argument(
        "--backend",
        choices=["auto", "metal", "cpu", "cuda", "vulkan"],
        default="auto",
        help="Compute backend (default: auto-detect)",
    )
    run_parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable all visualization (batch mode)",
    )
    run_parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live window; only save final image",
    )
    run_parser.add_argument(
        "--output",
        metavar="DIR",
        help="Override output directory from problem file",
    )
    run_parser.add_argument(
        "--show",
        action="store_true",
        help="Show final result interactively after optimization",
    )

    # --- validate ---
    val_parser = subparsers.add_parser(
        "validate", help="Validate a problem file without running"
    )
    val_parser.add_argument("problem", help="Path to YAML problem definition file")

    args = parser.parse_args()

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "validate":
        return _cmd_validate(args)
    else:
        parser.print_help()
        return 0


def _cmd_validate(args) -> int:
    from toptimization.problem import load_problem

    path = Path(args.problem)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        problem = load_problem(path)
        print(f"Problem '{problem.name}' is valid.")
        print(f"  Dimension   : {problem.dim}D")
        print(f"  Mesh        : {problem.nelx} x {problem.nely}" +
              (f" x {problem.nelz}" if problem.dim == 3 else ""))
        print(f"  Elements    : {problem.n_elem:,}")
        print(f"  DOFs        : {problem.n_dofs:,}")
        print(f"  Fixed DOFs  : {len(problem.fixed_dofs):,}")
        print(f"  Force DOFs  : {len(problem.force_dofs):,}")
        print(f"  Volume frac : {problem.volfrac}")
        print(f"  Filter      : {problem.filter_type} (rmin={problem.rmin})")
        print(f"  Max iter    : {problem.max_iter}")
        return 0
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1


def _cmd_run(args) -> int:
    from toptimization.problem import load_problem
    from toptimization.optimizer.simp import run

    path = Path(args.problem)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        problem = load_problem(path)
    except Exception as e:
        print(f"Error loading problem: {e}", file=sys.stderr)
        return 1

    # Override output directory if specified
    if args.output:
        problem.output_dir = Path(args.output)

    live_viz = not (args.no_viz or args.no_live)

    try:
        rho = run(
            problem=problem,
            backend=args.backend,
            live_viz=live_viz,
        )
    except Exception as e:
        import traceback
        print(f"Error during optimization: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    if args.show and not args.no_viz and problem.dim == 2:
        try:
            from toptimization.viz.plot2d import show_density
            show_density(rho, title=problem.name, block=True)
        except Exception as e:
            print(f"[warn] Could not show result: {e}")

    print(f"\nResults saved to: {problem.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
