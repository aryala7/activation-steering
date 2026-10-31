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

## Select a steering layer (Mistral)

First extract candidate layers **from training dialogues only**:

```sh
uv run --no-editable --extra models steer --model mistral --split train \
  --limit 300 --layers 8 12 16 20 24 --output-dir outputs/mistral-train
```

This `--limit` counts dialogues, not pairs. Inspect the printed number of pairs;
increase it if too few are eligible. Then generate on *different*, held-out
validation dialogues, using one unsteered baseline and each layer/strength:

```sh
uv run --no-editable --extra models steer sweep \
  --activations outputs/mistral-train/activations.pt \
  --limit 10 --strengths 0.5 1 --output-dir outputs/mistral-sweep
```

The sweep uses the raw mean difference at each layer and greedy generation.
It requires the training artifact to use Mistral. Start with a few turns: a
7B model on CPU needs substantial RAM and may generate very slowly. For the
actual layer comparison, increase `--limit` (for example to 100) and, after
the coarse sweep, extract and test neighboring layers in a separate run.
The `--max-new-tokens` default is 64; use the same value for every condition.

`predictions.jsonl` records responses and their layer/strength. `review.csv`
contains the same responses in shuffled order without showing the condition.
For **every** row, fill both columns with `0` or `1`:

- `constraint_ok`: 1 if the response makes no claim contradicting the active
  user constraints; 0 if it does. Check named hotels/restaurants against a
  database if claims about them cannot be judged from the text alone.
- `useful`: 1 if the response addresses the user's request with relevant
  information, or a justified explanation/clarification; 0 for evasive,
  unrelated, empty, or garbled responses.

A vague answer can be `constraint_ok=1, useful=0`. Check the dialogue history
and constraints while reviewing; the original MultiWOZ system response is not
shown or treated as an unquestionable reference. Then print the fractions for
each condition (sorted by the fraction satisfying both criteria):

```sh
uv run --no-editable --extra models steer rank \
  --predictions outputs/mistral-sweep/predictions.jsonl \
  --review outputs/mistral-sweep/review.csv
```

Choose a validation layer/strength that improves `both` over baseline without
a notable drop in usefulness. If none improves it, report that finding; do not
declare a layer best merely because its vector is large or its activations
separate well in a plot. Once chosen, run the same frozen configuration on the
MultiWOZ `test` split for the final result:

```sh
uv run --no-editable --extra models steer sweep \
  --activations outputs/mistral-train/activations.pt \
  --eval-split test --layers 12 --strengths 1 \
  --limit 100 --output-dir outputs/mistral-test
```

Replace `12` and `1` with your chosen values. Keep the generation settings and
review rules fixed. Do not use validation/test pairs to calculate vectors.
This is a turn-level evaluation with the real preceding
dialogue as context, not a full interactive rollout or the standard
delexicalized MultiWOZ Inform/Success evaluator.

## Code organization

`src/steer/main.py` is the single entry point and owns the extraction, sweep,
and review commands.
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
Overlong inputs raise an error instead of silently truncating context. During
the sweep the hook changes the final prompt position (which predicts the first
answer token) and each later decode step, without altering earlier positions
in the dialogue history. It is removed after each generation.

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
