"""Run the experiment: generate dialogue pairs, capture activations, save results."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from steer.contrastive_pairs import create_pairs, extract_slot_constraints, is_user
from steer.models.model import generate_response, make_vector
from steer.ontology import SLOT_VALUES
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_ID = "tuetschek/multi_woz_v22"
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def validation_turns(rows, limit):
    """Select assistant turns with supported active constraints, without using replies."""
    count = 0
    for row in rows:
        turns = row["turns"]
        constraints = {}
        for index in range(len(turns["utterance"])):
            if is_user(turns, index):
                constraints = {
                    key: values for key, values in
                    extract_slot_constraints(turns, index).items() if key in SLOT_VALUES
                }
                continue
            if not constraints or index == 0 or not is_user(turns, index - 1):
                continue
            messages = [
                {"role": "user" if is_user(turns, j) else "assistant",
                 "content": turns["utterance"][j]}
                for j in range(index)
            ]
            yield {
                "dialogue_id": row["dialogue_id"],
                "turn_id": turns["turn_id"][index],
                "messages": messages,
                "constraints": constraints,
            }
            count += 1
            if count >= limit:
                return


def sweep_main(argv):
    parser = argparse.ArgumentParser(description="Generate Mistral validation responses for a layer sweep")
    parser.add_argument("--activations", type=Path, required=True,
                        help="activations.pt created with --split train")
    parser.add_argument("--layers", nargs="+", type=int,
                        help="subset of layers in activations.pt (default: all)")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation",
                        help="Use test only after freezing the layer and strength")
    parser.add_argument("--limit", type=int, default=10, help="validation assistant turns")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    if not args.strengths or any(not 0 < x < float("inf") for x in args.strengths):
        parser.error("--strengths must contain finite positive numbers")
    if len(set(args.strengths)) != len(args.strengths):
        parser.error("--strengths must be unique")

    # The file is produced by this project's extraction command. Loading only
    # tensors and elementary types avoids executing arbitrary pickle objects.
    saved = torch.load(args.activations, map_location="cpu", weights_only=True)
    if saved.get("split") != "train" or saved.get("model_id") != MODELS["mistral"]:
        parser.error("Use train activations extracted from the Mistral model")
    saved_layers = saved["layers"]
    layers = args.layers if args.layers is not None else saved_layers
    if not layers or len(set(layers)) != len(layers) or set(layers) - set(saved_layers):
        parser.error("--layers must be unique and present in the extraction file")
    vectors = saved["mean_difference"]
    if vectors.ndim != 2 or vectors.shape[0] != len(saved_layers) or not torch.isfinite(vectors).all():
        parser.error("Invalid steering vectors in the extraction file")
    if any(vectors[saved_layers.index(layer)].norm().item() == 0 for layer in layers):
        parser.error("A selected steering vector is zero")

    rows = load_dataset(DATASET_ID, split=args.eval_split)
    turns = list(validation_turns(rows, args.limit))
    if not turns:
        parser.error("No eligible validation turns found")
    print(f"Loaded {len(turns)} {args.eval_split} turns. Loading Mistral on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["mistral"])
    model = AutoModelForCausalLM.from_pretrained(MODELS["mistral"], dtype=torch.float32)
    model.to(args.device).eval()
    revision = saved.get("model_revision")
    if revision and model.config._commit_hash and revision != model.config._commit_hash:
        parser.error("Mistral checkpoint revision differs from the extraction checkpoint")

    conditions = [(None, 0.0)] + [
        (layer, strength) for layer in layers for strength in args.strengths
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    review_rows = []
    with predictions_path.open("w", encoding="utf-8") as file:
        for turn in turns:
            for layer, strength in conditions:
                vector = None if layer is None else vectors[saved_layers.index(layer)]
                response = generate_response(
                    model, tokenizer, turn["messages"], args.max_new_tokens,
                    layer=layer, vector=vector, strength=strength,
                )
                record = {
                    "id": len(review_rows), "dialogue_id": turn["dialogue_id"],
                    "turn_id": turn["turn_id"], "layer": layer,
                    "strength": strength, "constraints": turn["constraints"],
                    "response": response, "eval_split": args.eval_split,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                review_rows.append({
                    "id": record["id"], "dialogue_id": turn["dialogue_id"],
                    "turn_id": turn["turn_id"],
                    "constraints": json.dumps(turn["constraints"], ensure_ascii=False),
                    "history": "\n".join(
                        f"{m['role']}: {m['content']}" for m in turn["messages"]
                    ),
                    "response": response, "constraint_ok": "", "useful": "",
                })
            print(f"Generated {turn['dialogue_id']} turn {turn['turn_id']}", flush=True)

    random.Random(42).shuffle(review_rows)  # Hide the setting/order while reviewing.
    review_path = args.output_dir / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Saved {predictions_path} and {review_path}")


def rank_main(argv):
    parser = argparse.ArgumentParser(description="Summarize reviewed layer-sweep responses")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    if not records:
        parser.error("No predictions to score")
    with args.review.open(encoding="utf-8", newline="") as file:
        reviews = list(csv.DictReader(file))
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != len(records) or len(reviews) != len(records):
        parser.error("Every prediction needs exactly one review row")
    scores = {}
    seen = set()
    for review in reviews:
        key = review["id"]
        if key not in by_id or key in seen:
            parser.error("Review ids do not match unique prediction ids")
        seen.add(key)
        if review["constraint_ok"] not in {"0", "1"} or review["useful"] not in {"0", "1"}:
            parser.error(f"Fill constraint_ok and useful with 0 or 1 for id {key}")
        record = by_id[key]
        condition = (record["layer"], record["strength"])
        result = scores.setdefault(condition, {"count": 0, "ok": 0, "useful": 0, "both": 0})
        ok, useful = int(review["constraint_ok"]), int(review["useful"])
        result["count"] += 1
        result["ok"] += ok
        result["useful"] += useful
        result["both"] += ok * useful
    if len({value["count"] for value in scores.values()}) != 1:
        parser.error("Conditions have unequal numbers of reviewed turns")
    print("layer,strength,n,constraint_ok,useful,both")
    for (layer, strength), value in sorted(
        scores.items(), key=lambda item: (-item[1]["both"] / item[1]["count"],
                                         -item[1]["useful"] / item[1]["count"])
    ):
        n = value["count"]
        print(f"{layer if layer is not None else 'baseline'},{strength:g},{n},"
              f"{value['ok']/n:.3f},{value['useful']/n:.3f},{value['both']/n:.3f}")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "sweep":
        return sweep_main(argv[1:])
    if argv and argv[0] == "rank":
        return rank_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10, help="Number of dialogues")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", default="mistral",
        help="Model alias (mistral or llama) or Hugging Face model ID; default: mistral",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[14])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    args.model = MODELS.get(args.model, args.model)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if min(args.layers) < 0 or len(set(args.layers)) != len(args.layers):
        parser.error("--layers must contain unique, nonnegative indices")

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
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
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
