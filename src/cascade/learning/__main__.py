"""Train, resume, evaluate and optionally export the reference tracking policy."""

import argparse

from .workflow import LearningRunConfig, run_learning


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="run directory (new/empty unless --resume)")
    parser.add_argument(
        "--resume", action="store_true", help="restore frozen settings and checkpoints"
    )
    parser.add_argument(
        "--steps", type=int, default=60, help="additional updates per seed; 0 evaluates"
    )
    parser.add_argument(
        "--export", action="store_true", help="serialize inference with the export extra"
    )
    parser.add_argument(
        "--reports", action="store_true", help="also write per-flight HTML inspectors"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--architecture", choices=("feedforward", "recurrent"))
    parser.add_argument("--task", choices=("tracking", "scheduled"))
    parser.add_argument("--horizon", type=int, dest="horizon_steps")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)
    options = {
        name: getattr(args, name)
        for name in LearningRunConfig.__dataclass_fields__
        if getattr(args, name) is not None
    }
    if args.resume and options:
        parser.error(
            "--resume restores its frozen configuration; omit training configuration options"
        )
    if "seeds" in options:
        options["seeds"] = tuple(options["seeds"])
    config = None if args.resume else LearningRunConfig(**options)
    return run_learning(
        args.output,
        config,
        steps=args.steps,
        resume=args.resume,
        export=args.export,
        reports=args.reports,
    )


if __name__ == "__main__":
    main()
