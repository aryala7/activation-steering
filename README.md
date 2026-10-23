# Steer

Generate contrastive system-response pairs from MultiWOZ 2.2 for activation-steering experiments.

## Run

Install the project and its dependencies with `uv sync`, then run:

```sh
uv run --no-editable steer --limit 10 --seed 42
uv run --no-editable python -m steer --limit 10 --seed 42
```

The default processes the first training dialogue. `--limit` counts dialogues,
not pairs. `--split` selects the dataset split. Output is JSON Lines on stdout;
the dataset must be downloadable or already cached.

Each record contains the shared `prompt`, `positive_response`,
`negative_response`, source dialogue/turn IDs, and the edited slot/span metadata.
The prompt includes preceding user and assistant turns but excludes the target
response and all subsequent turns.

## Structure

- `src/steer/contrastive_pairs.py`: pure pair generation and output types.
- `src/steer/ontology.py`: supported slot values.
- `src/steer/cli.py`: argument parsing, dataset loading, and JSON output.
- `src/steer/__init__.py`: public Python API (`create_pairs`, `ContrastivePair`).
- `src/steer/__main__.py`: module execution entry point.
- `tests/test_contrastive_pairs.py`: pair-generation regression tests.
- `tests/test_cli.py`: CLI validation and output tests.

## Pair selection

Only annotated system `Inform` and `Recommend` responses are candidates.
The original value must match the latest user-state constraint and the annotated
character span. A negative replaces that span with a value outside all allowed
values. User turns are never edited. Empty or cleared constraints are not reused.

Supported slots are hotel/restaurant area and price range. These are candidate
constraint-violation pairs, not guarantees that the complete original response
is factually correct. Review a sample before using them in experiments.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

On macOS, Python ignores `.pth` files marked hidden. If this causes a
`ModuleNotFoundError`, use `uv run --no-editable` as above. This installs the
package directly. Source files are included in the uv cache keys so changes
trigger a rebuild. Keep `--no-editable` on subsequent runs.
