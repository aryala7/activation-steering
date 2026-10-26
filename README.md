# Steer

One experiment pipeline: load MultiWOZ 2.2, generate contrastive response pairs,
run the model, and save activations and steering directions.

## Run

```sh
uv run --no-editable --extra models steer --limit 10 --layers 0 7 14
```

The command installs the required dependencies and downloads the dataset/model
when necessary. `--no-editable` avoids the macOS hidden `.pth` import issue.

Options:

- `--limit`: number of dialogues, default 10 (not number of pairs).
- `--split`: dataset split, default `train`.
- `--seed`: pair-generation random seed, default 42.
- `--model`: `mistral` (default), `llama`, or a full Hugging Face model ID.
- `--layers`: zero-based transformer block indices, default `14`.
- `--device`: `cpu` (default), `cuda`, or `mps`.
- `--output-dir`: results directory, default `outputs`.

Each run writes `pairs.jsonl` and `activations.pt` to the output directory,
overwriting earlier results there. Use a different output directory for each
experiment you want to keep.

## Models

| Alias | Hugging Face model |
| --- | --- |
| `mistral` (default) | `mistralai/Mistral-7B-Instruct-v0.3` |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` |

Run each model separately with its own output directory:

```sh
uv run --no-editable --extra models steer --model mistral --output-dir outputs/mistral
uv run --no-editable --extra models steer --model llama --output-dir outputs/llama
```

Each run loads the selected model and its matching tokenizer. Activations and
steering vectors are computed independently for each model.

## Code organization

`src/steer/main.py` is the single entry point and owns the complete workflow.
`__main__.py` delegates to it for `python -m steer`. The remaining modules
provide reusable functions and do not execute independently:

- `contrastive_pairs.py`: generate positive/negative system responses.
- `ontology.py`: supported hotel/restaurant area and price values.
- `models/model.py`: format chat inputs, capture activations, calculate vectors.

## What is captured

Each pair shares the same preceding user/assistant history. The original system
response is the positive candidate; a negative replaces one annotated value
with a value that contradicts the latest user-state constraint. These are
candidate pairs, not guarantees that the entire positive response is correct.

Both candidates are passed as assistant responses using the model's chat
template. The code records post-block residual states at the final response
content token, excluding the end marker. It does not generate text or apply
steering. Qwen2/Llama-style `model.model.layers` layouts are supported.
Overlong inputs raise an error instead of silently truncating context.

The activation file stores:

- Positive/negative activations and differences: `[pairs, layers, hidden_size]`.
- Raw mean differences and unit directions: `[layers, hidden_size]`.
- Source pairs and model, layer, token-position, dataset, split, and seed metadata.

Differences are positive minus negative. Raw means retain their magnitude;
unit directions are normalized separately for each layer.

## Tests

```sh
uv run --no-editable --extra models python -m unittest discover -s tests -v
```

Tests use small local models and mocked downloads, so no pretrained weights
are needed.
