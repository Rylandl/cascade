"""Fit and evaluate bounded aircraft calibrations from frozen flight-recording packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fitting import CalibrationConfig
from .workflow import evaluate_calibration, fit_flight_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit only the pack's fitting recordings")
    fit.add_argument("pack", type=Path, help="flight-pack manifest")
    fit.add_argument("output", type=Path, help="new or empty calibration directory")
    fit.add_argument("--config", type=Path, required=True, help="CalibrationConfig JSON file")
    evaluate = commands.add_parser("evaluate", help="replay a saved calibration on one split")
    evaluate.add_argument("pack", type=Path, help="the exact flight pack used for fitting")
    evaluate.add_argument("artifact", type=Path, help="calibration directory or metadata.json")
    evaluate.add_argument(
        "--split", choices=("fitting", "validation", "evaluation"), default="evaluation"
    )
    evaluate.add_argument("--output", type=Path, help="write replay results; otherwise print JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            config = CalibrationConfig.from_dict(json.loads(args.config.read_text()))
            artifact = fit_flight_pack(args.pack, config, output=args.output)
            optimizer = artifact.report["optimizer"]
            print("Calibration:", artifact.path)
            print("Optimizer:", "converged" if optimizer["success"] else "did not converge")
            print(optimizer["message"])
            print(
                f"Weighted data loss: {artifact.report['initial_data_loss']:.6g} -> "
                f"{artifact.report['final_data_loss']:.6g}"
            )
            sensitivity = artifact.report["identifiability"]
            print(f"Local sensitivity rank: {sensitivity['rank']}/{sensitivity['parameter_count']}")
            bound_hits = [
                row["path"] for row in artifact.report["parameters"] if row["bound_hit"] is not None
            ]
            if bound_hits:
                print("Parameters at bounds:", ", ".join(bound_hits))
            return 0 if optimizer["success"] else 3
        result = evaluate_calibration(
            args.artifact, args.pack, split=args.split, output=args.output
        )
        if args.output is None:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print("Replay results:", args.output)
        return 0
    except (ValueError, TypeError, OSError, FloatingPointError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
