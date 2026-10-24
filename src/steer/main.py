"""Run the experiment: generate dialogue pairs, capture activations, save results."""

import argparse
import json
import random
from pathlib import Path

from steer.contrastive_pairs import create_pairs
from steer.models.model import make_vector

DATASET_ID = "tuetschek/multi_woz_v22"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10, help="Number of dialogues")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layers", nargs="+", type=int, default=[14])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if min(args.layers) < 0 or len(set(args.layers)) != len(args.layers):
        parser.error("--layers must contain unique, nonnegative indices")

    from datasets import load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Generate contrastive pairs from the selected dialogues.
    print(f"Loading {DATASET_ID} ({args.split})...")
    rows = load_dataset(DATASET_ID, split=args.split)
    rng = random.Random(args.seed)
    pairs = []
    for row in rows.select(range(min(args.limit, len(rows)))):
        pairs.extend(create_pairs(row, rng))
    if not pairs:
        parser.error("No eligible pairs found. Try increasing --limit.")
    print(f"Created {len(pairs)} pairs.")

    # 2. Load the model once and capture both responses with their context.
    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(args.device).eval()
    result = make_vector(model, tokenizer, pairs, args.layers)
    result.update({
        "model_id": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "torch_version": str(torch.__version__),
        "dataset_id": DATASET_ID,
        "split": args.split,
        "seed": args.seed,
    })

    # 3. Save the source pairs and activation results together.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    activations_path = args.output_dir / "activations.pt"
    torch.save(result, activations_path)
    print(f"Saved pairs to {pairs_path}")
    print(f"Saved activations to {activations_path}")
