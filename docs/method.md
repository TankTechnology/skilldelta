# Method

For a fixed agent and supplied skill, `y0` and `y1` are a historical task's mean
success under Skip skill and Use skill. Its observed gain is `y1 - y0`.
The predictor transfers these labels through task-embedding similarity.

1. Exclude support records with the query's task ID.
2. Restrict support to the supplied-skill family, or use global support as
   specified by the protocol.
3. Select the nearest `k` tasks by cosine similarity. Ties preserve input order.
4. Average their gain labels with nonnegative cosine weights; use uniform
   weights if all selected weights are zero.
5. Use the skill only when the score is strictly above the threshold.
   Empty support returns Skip skill.

`predict_gain` accepts historical outcomes only. No outcome for a new query is
required. Same-ID exclusion also applies to the support-prevalence threshold.

[Protocol settings](../configs/protocols.json) preserve the paper's distinctions:
LogicBench uses the frequency of positive-gain labels with uniform aggregation,
which is a gain-sign score, not a signed increment estimate. MedCalc-Bench and
LogicBench use a support-prevalence threshold; other configurations use zero.

`predict_skill_success` is the skill-on-only ranking control and accepts no
Skip skill outcomes. Other controls in `skilldelta/baselines.py` include lexical
relevance and leave-one-out family mean gain.

The Harness deployment plugin uses uniform neighbor aggregation; it is
documented separately from the benchmark configurations.
