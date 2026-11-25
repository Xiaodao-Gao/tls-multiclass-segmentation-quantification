"""One command-line interface; no implicit data or weight downloads."""
import argparse

import torch

from . import __version__


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tlsseg")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train and select by validation only")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=positive_int)
    train.add_argument("--resume", help="Resume a repository last.pt checkpoint")
    train.add_argument("--native-checkpoint", help="Optional local plain SMP U-Net state_dict")
    train.add_argument("--max-train-batches", type=positive_int, help="Partial smoke run, marked in outputs")
    train.add_argument("--max-val-batches", type=positive_int, help="Partial smoke run, marked in outputs")
    evaluation = sub.add_parser("evaluate", help="Evaluate one frozen checkpoint without tuning")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--split", choices=("train", "val", "test"), required=True)
    evaluation.add_argument("--output", required=True, help="Report JSON path")
    evaluation.add_argument("--external-cohort", action="store_true", help="Explicit new cohort; rejects patient IDs in original manifest")
    prediction = sub.add_parser("predict", help="Export per-tile probabilities and class-index PNGs")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--split", choices=("train", "val", "test", "predict"))
    prediction.add_argument("--output", required=True)
    for command in (train, evaluation, prediction):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
        command.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    for command in (evaluation, prediction):
        command.add_argument("--batch-size", type=positive_int, default=4)
        command.add_argument("--num-workers", type=nonnegative_int, default=0)
    stitch = sub.add_parser("stitch", help="Fuse probabilities on a declared grid and quantify area")
    stitch.add_argument("--layout", required=True, help="Layout JSON with coordinates and scale metadata")
    stitch.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "threads"):
        torch.set_num_threads(args.threads)
    if args.command == "train":
        from .engine import train as run
    elif args.command == "evaluate":
        from .engine import evaluate as run
    elif args.command == "predict":
        from .predict import predict as run
    else:
        from .predict import stitch as run
    run(args)
