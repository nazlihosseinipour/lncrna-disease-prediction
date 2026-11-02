# Prompt to start the Claude Opus orchestration session

You are the supervising and decision-making agent for this task. Your role is to understand the
whole experiment, design the work breakdown, assign bounded work to lower-cost subagents, review
their evidence, resolve conflicts, integrate changes, and decide whether each acceptance gate has
actually passed.

Do not spend the Opus context doing routine repository searches, mechanical inventories, or
isolated implementation that a subagent can perform. Delegate those tasks to Sonnet or Haiku
subagents where available. Keep architecture, scientific protocol decisions, risk decisions,
integration review, and final verification with Opus.

## Authoritative instructions

Read these files in order before dispatching work:

1. `Reports/FELIPE_IMPLEMENTATION_AND_AUDIT_PLAN_2026-09-27.md`
2. `Reports/CLAUDE_CONTINUATION_HANDOFF_2026-09-27.md`
3. `GATES.md`
4. `results/felipe_basics_2026-09-27/inputs/input_summary.json`
5. the manifests under `config/felipe_basics_*.csv`

The file `/Users/nazlihosseinipour/Downloads/Pasted text.md` is an older, broad audit request. You
may use its questions as a historical checklist when relevant, but it is not the active scope. It
must not expand this task into all feature families, binary experiments, concatenation studies,
or publication figures. If it conflicts with the current Felipe plan, the current plan wins.

Do not assume earlier Claude reports are correct. Code, raw inputs, manifests, prediction files,
hashes, and logs are the evidence.

## Active objective

Produce a scientifically and technically verified implementation for:

- V1→V1;
- V2→V2;
- V1→V2;
- V2→V1;
- RF, IPCARF, and RFLDA in every direction.

Use the identical reconstructed 542 sequence-only features for both datasets. Within-version runs
keep all disease columns in the declared population. Transfer uses only the ordered common disease
space, currently 73 diseases. Keep the paper-comparison evaluation separate from the primary
clean evaluation.

Do not use “V2 must score higher than V1” as a correctness condition.

## How to orchestrate

Before editing, create a short task table with:

- task name;
- exact scope;
- files the task may edit;
- dependencies;
- assigned subagent/model;
- required evidence;
- status.

Give every agent exclusive file ownership. Run independent read-only or non-overlapping tasks in
parallel. Do not let two agents edit the same file. Give agents the minimum context needed plus the
relevant acceptance gates.

Suggested first wave:

### Agent A — V2 population and data provenance

Read-only first. Determine whether authoritative local sequences exist for the 1,208 V2 rows that
currently lack sequence data. Verify V1/V2 RNA IDs, sequence provenance, duplicates, and the
current exclusion population. Return an evidence table and proposed exclusion ledger. Do not
download, synthesize, or zero-fill sequences without an explicit Opus decision.

Expected ownership if a file is needed:

- a new V2 missing-sequence audit or exclusion CSV only.

### Agent B — model fidelity and dirty-code audit

Inspect the working-tree changes and committed versions of `IPCARF/ipcarf.py` and
`RFLDA/rflda.py`. Determine which exact code implements the paper profile, which local changes are
unvalidated, and what source hashes must be recorded. Run only focused, inexpensive tests.

Expected ownership:

- a new model-fidelity audit file and focused model tests only;
- do not edit the model implementations in the first pass.

### Agent C — execution/rescoring design audit

Inspect `run_inductive_within_cv.py`, `run_felipe_strict_transfer.py`, prediction formats,
summaries, and verifier behavior. Produce an implementation proposal for atomic transfer resume,
corruption rejection, one-fit/two-score rescoring, threshold-mode-safe keys, and the 73-label
within baselines.

Expected ownership in the first pass:

- a new technical design/audit file only;
- no long model execution.

## Opus integration checkpoint after the first wave

Read and independently verify every agent report against the repository. Then decide:

1. the declared V2 population;
2. the exact frozen model sources for the paper profile;
3. the prediction and protocol format;
4. the implementation split and exclusive file ownership for the next wave;
5. which gates can become runnable checks.

Do not start full training at this checkpoint.

## Suggested implementation wave

After the checkpoint, delegate separate owned tasks such as:

- transfer persistence/resume implementation and tests;
- prediction rescoring implementation and tests;
- manifest/verifier updates;
- V2 exclusion ledger and population naming;
- 73-label within-baseline manifest preparation.

Require each implementation agent to return:

- files changed;
- commands run and exact results;
- assumptions;
- unresolved risks;
- a small reproducible probe;
- confirmation that no long run or push occurred.

Opus must review diffs, rerun the smallest decisive checks, and integrate the pieces. A subagent's
“done” message is not acceptance evidence.

## Execution rules

- Begin with read-only inspection and small tests.
- Do not start full cross-validation until all P0 items and probe gates pass.
- Save raw probabilities once; do not refit merely to change thresholds.
- Never select clean thresholds from test or transfer-target labels.
- Never append disease labels to X.
- Never align X and Y by row position.
- Never silently filter disease columns.
- Never silently discard the 1,208 missing-sequence V2 rows.
- Never repair old outputs in place.
- Do not trust a summary CSV without its predictions, IDs, hashes, and completion evidence.
- Do not push, publish, broadly commit, run `git add -A`, or clean the existing dirty tree.

## Communication and stopping rule

Keep a concise decision log explaining protocol choices and rejected alternatives. Tell me when a
decision changes the scientific population or evaluation meaning.

Stop before expensive full runs and report:

1. what the agents found;
2. Opus's verified decisions;
3. files changed;
4. tests and probes passed;
5. gates still open;
6. the exact proposed full-run commands and expected runtime/resource needs.

Do not claim completion until every required gate in `GATES.md` has evidence.
