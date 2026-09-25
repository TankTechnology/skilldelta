# Generate data locally

Collection code is included; benchmark questions, skill corpora, traces, and
vectors are not. Obtain inputs under their original terms and keep them local.

## Original collection adapters

SR-Agents adapters expect the upstream checkout at `experiments/data/SR-Agents`:

```bash
git clone https://github.com/oneal2000/SR-Agents.git experiments/data/SR-Agents
git -C experiments/data/SR-Agents checkout 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f
```

Install its dependencies and follow its benchmark asset setup. Install the API
client separately (`pip install openai`). The upstream corpus and external
ToolQA assets are required locally; our scripts do not fetch them implicitly.

| Benchmark | Prepare inputs | Collect paired executions |
| --- | --- | --- |
| ToolQA | `prepare_sra_toolqa_all.py` | `run_sra_toolqa.py` |
| MedCalc-Bench | `prepare_sra_medcalc.py` | `run_sra_medcalc.py` |
| BigCodeBench | `prepare_sra_bigcode_full.py` | `run_sra_bigcode.py`, then `evaluate_sra_bigcode.py` |
| LogicBench | `prepare_sra_bench.py --dataset logicbench` | `run_sra_logicbench.py` |
| SpreadsheetBench | SkillOpt setup and local split files | `run_eval.py` |

Scripts live in `experiments/`. Preparation uses upstream annotations without
model calls. Their original scopes remain explicit: the LogicBench helper makes
pilot/development/audit panels, whereas ToolQA and BigCodeBench helpers make full
inventories. These helpers do not recreate every historical split on their own.

The four `run_sra_*.py` programs print a plan unless `--execute` is supplied.
Set `--repetitions 3` for three executions per condition. `no_skill` means Skip
skill; the Use skill arm is `oracle_skill` or `all_gold_skills`. Internal names
are retained for artifact compatibility. Check each runner's `--help` for the
supported panel and arm names.

For example, this prepares MedCalc inputs and prints a **dry-run plan**:

```bash
python experiments/prepare_sra_medcalc.py
python experiments/run_sra_medcalc.py \
  --manifest experiments/data/sra_bench/medcalc_seed20260828/manifest.json \
  --panel experiments/data/sra_bench/medcalc_seed20260828/support.json \
  --panel-name support \
  --skills experiments/data/sra_bench/medcalc_seed20260828/selected_skills.json \
  --output experiments/outputs/medcalc/support.jsonl \
  --model qwen-turbo --repetitions 3
```

Live runs require explicit call/token/cost/quota caps and durable budget/ledger
paths. Credentials come from environment variables named by `--api-base-env`
and `--api-key-env`; never place their values in tracked configuration.
`api_budget.py` retains the original gateway's `/api/usage/token/` quota contract.
That contract is provider-specific. Use a compatible gateway or adapt the quota
query to your provider while retaining call/token accounting. Historical prices
and default budgets describe old run settings and need review for a new run.

Use one writer per output directory. Resume with the same inputs, configuration,
ledger, and output paths. Preserve terminal errors rather than silently retrying
until success. BigCodeBench evaluation requires Linux, bubblewrap, and a prepared
Python evaluation environment.

## SpreadsheetBench

`run_eval.py` expects [SkillOpt](https://github.com/microsoft/SkillOpt) at
`third_party/SkillOpt`, pinned to `0389ace56339988e16ca5ddab36f0978776fe9b0`.
Install its dependencies and obtain SpreadsheetBench assets separately. Edit
`experiments/configs/spreadsheetbench_eval.yaml` for your local paths, target
model and worker limits. The wrapper patches the pinned config loader and
records direct SDK token usage.

Unlike the SRA runners, this wrapper starts evaluation when invoked. Run the
same frozen tasks with an empty skill and the supplied skill, keeping each
repeat/condition separate. Set `SKILLDELTA_MAX_API_CALLS` and
`SKILLDELTA_MAX_TOTAL_TOKENS` before invocation. Credentials remain in the
`TARGET_OPENAI_COMPATIBLE_API_KEY` and `TARGET_OPENAI_COMPATIBLE_BASE_URL`
environment variables.

## Representations

`build_medcalc_embeddings.py` retains the original question-only API encoder,
including manifest checks, batching, checkpoints and budgets. It prints a plan
by default; `--live` requires a compatible quota endpoint.

For a new experiment with a local encoder:

```bash
python -m pip install sentence-transformers
python scripts/encode_tasks.py \
  --input local_data/questions.json --model /path/to/local/encoder \
  --output local_data/embeddings.npz --batch-size 16 --device cpu
```

The portable utility reads task text only and uses a local checkpoint, without
remote model code or automatic downloads. Document the checkpoint revision and
settings; changing the encoder defines a new experimental setting.
