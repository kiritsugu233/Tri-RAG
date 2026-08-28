# Prompt for the next implementation agent

Work in this repository on branch `codex/calibrated-tri-predict-v2`.

Raw Tri-Predict v1 is complete and immutable at tag
`raw-tri-predict-v1-terminal-negative` (`fb09c00`). Its real SciFact result is a
terminal negative baseline: pilot LID is biased low, while the analytic
LID-to-budget map remains severely overconservative even under oracle LID.

Your task is to implement Pilot-Distance Calibrated Tri-Predict v2 without
editing v1 behavior or reusing SciFact cert/test for selection.

Before editing, read `AGENT_CALIBRATED_TRI_PREDICT.md` completely and follow it.
Then read every file listed in its “Read before editing” section. Start with the
network-free first implementation pass only. Do not download data, run an LLM,
or open any protected real split in the first pass.

The desired positive result is a hypothesis, not a required outcome. Preserve
all failures and maintain strict cal/tune/cert/latency/test separation.
