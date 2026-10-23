"""Command-line interface for generating MultiWOZ response pairs."""

import argparse
import json
import random
from collections.abc import Sequence

from steer.contrastive_pairs import create_pairs

DATASET_ID = "tuetschek/multi_woz_v22"


def main(argv: Sequence[str] | None = None) -> None:
    """Write one JSON object per pair to standard output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit", type=int, default=1,
        help="Number of dialogues (default: first row).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")

    # Keep dataset dependencies out of pure pair generation and CLI help.
    from datasets import load_dataset

    rows = load_dataset(DATASET_ID, split=args.split)
    rng = random.Random(args.seed)
    for row in rows.select(range(min(args.limit, len(rows)))):
        for pair in create_pairs(row, rng):
            print(json.dumps(pair, ensure_ascii=False))
