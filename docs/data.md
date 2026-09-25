# Local input formats

No benchmark records or empirical results are bundled. Store downloaded and
generated files under `local_data/`, `experiments/data/`, or
`experiments/outputs/`; these directories are excluded from Git and releases.

## Evaluation panel

Supply a JSON object with a `rows` list. Each row describes one task:

| Field | Meaning |
| --- | --- |
| `id` | Unique task identifier |
| `family` | Supplied-skill support group |
| `y0`, `y1` | Mean success under Skip skill and Use skill, each in [0, 1] |
| `tokens0`, `tokens1` | Mean reported execution tokens in the two conditions |

Compute means from declared repeats of the same task and model. Keep error
handling and denominators fixed. Do not treat a missing execution as a measured
failure or mix models in one history.

Embeddings use an NPZ file with `ids` (strings) and `vectors` (a finite N × D
array). IDs and order must exactly match the panel. Zero vectors are rejected.
Only task text enters the encoder; outcomes remain separate support labels.

## Encoding input

`scripts/encode_tasks.py` accepts a JSON list of objects with `id` and `question`.
It writes the NPZ interface above using a local Sentence Transformers checkpoint.
This portable utility is for new experiments: document the checkpoint and
settings rather than assuming it reproduces a different encoder.

## Collection and plugin formats

The original collection adapters consume SR-Agents instance lists, skill-corpus
lists, and hashed manifests. Their native JSONL output records each task,
condition, and repeat. BigCodeBench generation and evaluation are separate.
See [data generation](data-generation.md) before collecting results.

The Harness router has a separate support-index interface documented in the
[plugin README](../plugins/dsh-skilldelta/README.md). Its examples are synthetic.
