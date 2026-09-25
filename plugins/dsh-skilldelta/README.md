# SkillDelta for DeepSeek Harness

Enable a supplied skill before the first model step, using gains from historical
paired executions. The plugin exposes route/status tools and can inject the skill
automatically when predicted gain exceeds a threshold.

## 1. Try it offline

From the repository root, using Python 3.10+:

```bash
make plugin-demo
```

This synthetic example needs no Harness installation, model, or embedding API.
It returns `predicted_gain: 1.0`, `enable_skill: true`, and two neighbors.
For a Skip skill decision:

```bash
python plugins/dsh-skilldelta/scripts/route.py route \
  --support plugins/dsh-skilldelta/examples/support.json \
  --query-index plugins/dsh-skilldelta/examples/queries.json \
  --task-id demo-skip --k 2
```

The query index contains only IDs and vectors. A query ID is excluded from the
support neighbors even if it appears in the support file.

## 2. Connect to Harness

Install the pinned JavaScript dependencies:

```bash
npm ci --prefix plugins/dsh-skilldelta
```

Edit [examples/profile.patch.yml](examples/profile.patch.yml), replacing its
absolute paths with your checkout and your own support/skill files. Load it into
an existing configured Harness profile:

```bash
dsh --profile headless --patch /absolute/path/profile.patch.yml --json "Your task"
```

The profile supplies the target model and provider configuration. For a synthetic
integration demo, set `SKILLDELTA_TASK_ID=demo-help` and use the example files.
Running `dsh` calls the configured model and can incur provider charges;
`make plugin-demo` does not.

For an unseen task without a stored vector, configure an OpenAI-compatible
embedding service through environment variables:

```bash
export SKILLDELTA_EMBEDDING_BASE_URL="https://your-embedding-provider.example/v1"
export SKILLDELTA_EMBEDDING_MODEL="your-support-encoder"
# Set SKILLDELTA_EMBEDDING_API_KEY through your normal secret-management workflow.
```

Use the support index's encoder and dimension. The support outcomes and supplied
skill should refer to the intended agent/skill condition. For query-index replay,
set `SKILLDELTA_TASK_ID` separately for each process.

## 3. Supply historical evidence

A frozen support file lists eligible tasks:

```json
{
  "schema": "skilldelta-support-v1",
  "embedding_model": "your-support-encoder",
  "tasks": [
    {"id": "history-1", "embedding": [1.0, 0.0], "d": 1.0},
    {"id": "history-2", "embedding": [0.0, 1.0], "d": -1.0}
  ]
}
```

`d` is the Use skill success outcome (or repeat mean) minus the Skip skill
outcome. Select the skill/family scope when preparing the file; this adapter
does not infer a family from arbitrary task text.

The adapter ranks by cosine similarity, averages the signed gains of up to `k`
neighbors **uniformly**, and uses the skill when gain is strictly greater than
`threshold`. Similarity ties are ordered by task ID. The paper's default uses
clipped-cosine weights and benchmark-specific settings, implemented separately
in [the core predictor](../../skilldelta/predictor.py). This plugin is a runtime
integration example, not the evaluator behind the paper tables.

## Configuration

| Setting | Purpose |
|---|---|
| `support`, `skill` | Frozen support JSON and skill document |
| `queryIndex` | Optional outcome-free vectors for replay |
| `pythonCmd` | Python executable; default `python3` |
| `k`, `threshold` | Neighbor count and strict threshold; defaults 10 and 0 |
| `autoRoute` | Inject at the first step of each turn; default true |
| `failMode` | On routing errors: `always-on`, `always-off`, or `reject` |
| `routeLog` | Optional append-only JSONL decision log |
| `taskIdEnv` | Task-ID environment variable; default `SKILLDELTA_TASK_ID` |

Failure modes apply **only if routing fails**; they are not fixed policies.
The original default is `always-on`. The example uses `reject` to expose setup
errors. A successful negative-gain decision skips the skill in every failure mode.
Logs include task text and selected neighbors; choose storage and retention accordingly.

`skilldelta_route` inspects a decision without injection; `skilldelta_status`
reports configuration and file digests. The automatic hook injects at `step=1`.

## Tests and compatibility

```bash
make test-plugin
```

Python tests cover target exclusion, query-only replay, thresholds, and invalid
inputs. Node tests use the pinned Harness packages, real Python router, and
synthetic fixtures to check injection, skipping, and all error policies.

The original integration targeted official Harness commit
`ddefc45fbc7f8e46dd73185e68295696d1297887`. The package lock pins dependencies
used by the local hook tests. See the [official Harness project](https://github.com/deepseek-ai/deepseek-harness)
for profile setup; check compatibility before using later Harness versions.

[中文说明](README.zh.md) · [Paper reproduction](../../docs/reproduction.md)
