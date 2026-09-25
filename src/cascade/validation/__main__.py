"""Run: JAX_ENABLE_X64=1 python -m cascade.validation OUTPUT."""

import argparse

from .campaign import fixtures, run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--models", nargs="+", choices=list(fixtures()))
    parser.add_argument(
        "--smoke", action="store_true", help="short workflow check; not the full maneuver campaign"
    )
    args = parser.parse_args()
    result = run_campaign(args.output, models=args.models, smoke=args.smoke)
    print(f"Passed: {result['passed']}; report: {args.output}/summary.md")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
