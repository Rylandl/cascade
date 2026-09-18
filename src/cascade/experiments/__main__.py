"""Run a saved scenario manifest with the packaged baseline policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from cascade.experiments import Experiment, autotuned_policy, run_experiment, trim_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="saved experiment manifest JSON")
    parser.add_argument("output", type=Path, help="new or empty results directory")
    parser.add_argument("--policies", default="trim,cascade", help="comma-separated trim,cascade")
    parser.add_argument(
        "--split", choices=("training", "validation", "evaluation"), default="evaluation"
    )
    parser.add_argument("--no-reports", action="store_true", help="omit offline HTML inspectors")
    args = parser.parse_args(argv)
    factories = {"trim": trim_policy, "cascade": autotuned_policy}
    names = [name.strip() for name in args.policies.split(",")]
    if any(name not in factories for name in names) or len(set(names)) != len(names):
        parser.error("--policies must list unique names from trim,cascade")
    try:
        experiment = Experiment.load(args.manifest)
        result = run_experiment(
            experiment,
            [factories[name]() for name in names],
            args.output,
            split=args.split,
            reports=not args.no_reports,
        )
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"Experiment {result['experiment_sha256']}")
    print(f"Scored {len(result['episodes'])} episodes; results: {args.output / 'results.json'}")
    return result


if __name__ == "__main__":
    main()
