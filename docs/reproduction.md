# Evaluate locally held data

This repository ships code, not the paper's execution artifacts. Tests verify
implementation behavior on synthetic inputs; reproducing numerical tables also
requires the corresponding data.

After [collecting paired records](data-generation.md), prepare the panel and
embeddings described in [input formats](data.md), then run:

```bash
python scripts/evaluate.py \
  --panel local_data/panel.json \
  --embeddings local_data/embeddings.npz \
  --benchmark ToolQA \
  --output results/evaluation.json
```

`--benchmark` selects the predictor configuration, not a bundled dataset.
The evaluator recomputes leave-one-task-out predictions and reports gain-sign
classification, selected-policy success and tokens, Always-skip/Always-use,
matched-rate random activation, and a skill-on-only ranking control.
It reads no archived predictions or expected result values.

For a new query, use `predict_gain` as shown in
[the example](../examples/predict_before_execution.py). Keep query outcomes
unavailable until decisions have been recorded.

Execution tokens exclude encoder and routing costs unless explicitly added;
`--mean-router-tokens` accepts measured per-task routing overhead. Fixed-panel
LOTO metrics do not replace prospective evaluation on independent tasks.
