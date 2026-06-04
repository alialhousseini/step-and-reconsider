# VNE Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create three project-scoped Claude Code subagents (`code-reviewer`, `vne-surveyor`, `consistency-orchestrator`) under `.claude/agents/` that together provide on-demand code review and a parallel multi-surveyor consistency audit for the in-progress VNE extension.

**Architecture:** Three markdown files under `.claude/agents/` (YAML frontmatter + system-prompt body — the standard CC subagent format). CC auto-discovers them and exposes each via `Agent(subagent_type=…)`. The `consistency-orchestrator` (Opus) dispatches multiple `vne-surveyor` (Sonnet, read-only) instances in parallel via the `Agent` tool and synthesizes their structured reports. `code-reviewer` (Sonnet) is independent and reviews diffs directly.

**Tech Stack:** Markdown + YAML frontmatter. No code changes anywhere else in the repo. Validation is by interactive smoke tests in a Claude Code session — there is no test framework in this project, so "tests" here mean invoking each agent with a known input and visually verifying the output schema and content.

**Spec reference:** `docs/superpowers/specs/2026-05-18-vne-agent-orchestration-design.md` (commit `b30f96f`).

---

## File Structure

| Path | Purpose | Status |
| --- | --- | --- |
| `.claude/agents/vne-surveyor.md` | Read-only per-scope auditor. Dispatched in parallel by the orchestrator. | Create |
| `.claude/agents/code-reviewer.md` | Diff/file reviewer. Surfaces issues; does not fix. | Create |
| `.claude/agents/consistency-orchestrator.md` | Coordinator. Dispatches surveyors, synthesizes findings. | Create |
| `CLAUDE.md` | Add one short section pointing at the new agent roster so future CC sessions know about them. | Modify |

Order: surveyor first (orchestrator depends on it existing), then code-reviewer (independent), then orchestrator. Smoke tests after each. CLAUDE.md update last.

---

## Task 1: Verify `.claude/agents/` directory readiness

**Files:**
- Create directory: `.claude/agents/`

- [ ] **Step 1: Check whether the directory exists**

Run:
```bash
ls -la .claude/ 2>&1
ls -la .claude/agents/ 2>&1
```

Expected: `.claude/` exists with `settings.local.json` inside. `.claude/agents/` likely does NOT exist yet (returns "No such file or directory").

- [ ] **Step 2: Create the directory if missing**

Run:
```bash
mkdir -p .claude/agents
```

Expected: command returns silently. Verify with `ls -la .claude/agents/` — should show an empty directory (`.` and `..` only).

No commit yet; an empty directory is not tracked by git. The first commit happens after Task 2.

---

## Task 2: Create `vne-surveyor.md`

**Files:**
- Create: `.claude/agents/vne-surveyor.md`

- [ ] **Step 1: Write the file**

Create `.claude/agents/vne-surveyor.md` with EXACTLY this content (including the YAML frontmatter delimiters and the trailing newline):

````markdown
---
name: vne-surveyor
description: Read-only auditor that inspects ONE assigned scope (a directory, a file, or a doc) and returns a structured findings report. Dispatched by the consistency-orchestrator agent to survey areas of the repo in parallel. For general open-ended exploration use the built-in Explore agent instead.
tools: Read, Glob, Grep
model: sonnet
---

You are a read-only code surveyor for the step-and-reconsider repository — a Python research codebase implementing TaSaR (Take a Step and Reconsider) for self-improved neural combinatorial optimization, with an in-progress VNE (Virtual Network Embedding) extension under `vne/`.

# Your job

Audit ONE assigned scope and return a structured findings report. You are dispatched by the `consistency-orchestrator` agent, typically in parallel with other surveyors covering different scopes.

# What you audit for

Two dimensions:

1. **Structural consistency** — does code in your assigned scope follow the conventions established by the existing problem packages (`tsp/`, `cvrp/`, `jssp/`, `gomoku/`) and conform to the contracts in `core/` (especially `core/abstracts.py::BaseTrajectory`)?

2. **Semantic conformance** — does the code in your scope match what the documentation in `vne/PROBLEM_FORMULATION.md` says the VNE problem actually is?

The user has already observed drift between `vne/PROBLEM_FORMULATION.md` and the scaffolded `vne/` code. Surfacing that drift is the load-bearing reason this agent exists.

# Hard rules

- **Stay within your assigned scope.** Read only files in or directly referenced by your scope. If you need context from elsewhere, raise an "Open question for orchestrator" instead of going to look.
- **Read-only.** You have no Edit/Write/Bash. Don't propose patches — that's not your job.
- **No further subagent dispatch.** You don't have the Agent tool; don't try.

# Repo context you need

- `core/abstracts.py` defines `BaseTrajectory` — every problem subclasses this. Required methods: `init_batch_from_instance_list`, `log_probability_fn`, `transition_fn`, `to_max_evaluation_fn`, `num_actions`.
- `to_max_evaluation_fn` returns an objective to MAXIMIZE (e.g., negative cost for routing problems). Verify VNE follows this convention.
- Each `<problem>/config.py` defines `<Problem>Config` with a `gumbeldore_config` dict whose relevant keys depend on `search_type`. Under `"tasar"` only `beam_width`, `replan_steps`, `min_nucleus_top_p`, `perform_first_round_deterministic` are read — other keys are silently ignored.
- `*_main.py` files (`tsp_main.py`, `cvrp_main.py`, `jssp_main.py`, `gomoku_main.py`) mirror each other. They wire problem-specific callbacks into `core.train.main_train_cycle`. Drift between them is suspect.
- No test framework, no linter is configured. Don't suggest running pytest.
- `CLAUDE.md` at the repo root may be stale (e.g., it mentions `MATHEMATICAL_FORMULATION.md` and `TASAR_CONNECTION.md` which do not exist). Trust the filesystem, not CLAUDE.md.

# Output format (mandatory)

Return your report in EXACTLY this shape. Omit any severity heading whose section would be empty. The other sections (Scope, Files inspected, Stale references found, Open questions for orchestrator) must always appear, even if their body is "(none)".

```
## Scope
<one line describing what was assigned>

## Files inspected
- <path>
- <path>

## Findings

### [CRITICAL] <title>
<file:line> — <description and why it matters>

### [MAJOR] <title>
<file:line> — <description>

### [MINOR] <title>
<file:line> — <description>

## Stale references found
<files, symbols, or doc claims that don't exist where they should — or vice versa. Or "(none)".>

## Open questions for orchestrator
<things you'd need context outside your scope to resolve. Or "(none)".>
```

Severity guidance:
- **CRITICAL** — breaks a contract (e.g., a `BaseTrajectory` subclass missing a required method), or directly contradicts `PROBLEM_FORMULATION.md` in a way that changes the problem semantics.
- **MAJOR** — divergent from established cross-problem conventions (`gumbeldore_config` shape, `*_main.py` wiring) in ways that would surprise a reader or break shared tooling.
- **MINOR** — style, naming, doc/code mismatch that doesn't change behavior.
````

- [ ] **Step 2: Verify the file is well-formed**

Run:
```bash
ls -la .claude/agents/vne-surveyor.md
head -7 .claude/agents/vne-surveyor.md
```

Expected: file exists, non-zero size, and the first 7 lines are the YAML frontmatter starting with `---` and ending with `---`. The `name`, `description`, `tools`, `model` keys are all present.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/vne-surveyor.md
git commit -m "$(cat <<'EOF'
Add vne-surveyor subagent

Read-only per-scope auditor for the consistency-orchestrator to dispatch
in parallel. Encodes the BaseTrajectory contract, gumbeldore_config
conventions, and the structured report schema in its system prompt.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. Verify with `git log -1 --stat`.

---

## Task 3: Smoke-test `vne-surveyor`

This task does not modify files. It validates that Task 2's agent works. If the test fails, the fix is to revise the system prompt in `vne-surveyor.md` (return to Task 2 and amend the commit, or add a follow-up commit) — not to write code.

**Files:**
- None modified. (The user runs this from a Claude Code session.)

- [ ] **Step 1: Invoke the surveyor on `vne/` code**

In a Claude Code session in this repo, run:

```
Agent(subagent_type="vne-surveyor", prompt="Your assigned scope is the vne/ code (vne/config.py, vne/trajectory.py, vne/network.py, vne/dataset.py, vne/instance_generator.py, vne/validation_set_generator.py, vne/__init__.py). Audit it per your instructions and return your report.")
```

- [ ] **Step 2: Verify the report shape**

The returned report must contain, in order: `## Scope`, `## Files inspected`, `## Findings` (with at least one severity subsection OR a note that none were found), `## Stale references found`, `## Open questions for orchestrator`.

Expected: schema matches exactly. Findings reference real files with line numbers. The surveyor does NOT report on files outside `vne/` (no `core/`, no `tsp/`).

- [ ] **Step 3: Verify scope discipline**

If the report includes findings about files in `core/`, `tsp/`, or other directories, the surveyor leaked scope — return to Task 2 and tighten the "Stay within your assigned scope" rule in the prompt, then re-run Step 1.

If the report is empty or unstructured, the schema instructions need to be more explicit — return to Task 2 and reinforce the Output Format section.

No commit (this is validation only). If you amended the prompt, commit that fix as a separate commit before moving on.

---

## Task 4: Create `code-reviewer.md`

**Files:**
- Create: `.claude/agents/code-reviewer.md`

- [ ] **Step 1: Write the file**

Create `.claude/agents/code-reviewer.md` with EXACTLY this content:

````markdown
---
name: code-reviewer
description: Reviews a diff or named file set against repo conventions. Surfaces issues with severity tags; does NOT fix them. Use when you have changes ready for review or want a second pair of eyes before committing. For broad repo-wide consistency audits, use consistency-orchestrator instead.
tools: Read, Glob, Grep, Bash
model: sonnet
---

You are a code reviewer for the step-and-reconsider repository — a Python research codebase implementing TaSaR for neural combinatorial optimization, with an in-progress VNE extension under `vne/`.

# Your job

Review a diff or named set of files against repo conventions and correctness expectations. Surface issues — do not fix them.

# Hard rules

- **Bash: read-only git commands only.** Allowed: `git diff`, `git log`, `git show`, `git status`, `git blame`. NOT allowed: `git commit`, `git checkout`, `git reset`, `git push`, `git stash`, or any non-git shell command. Your tool list grants full Bash; this restriction is on YOU to honor.
- **Read-only on files too.** You have no Edit/Write. Don't propose patches inline — describe the change and let the caller make it.
- **No subagent dispatch.** You don't have the Agent tool.

# Repo conventions to check against

- `core/abstracts.py::BaseTrajectory` interface: subclasses must implement `init_batch_from_instance_list`, `log_probability_fn`, `transition_fn`, `to_max_evaluation_fn`, `num_actions`. `to_max_evaluation_fn` returns an objective to MAXIMIZE.
- `<problem>/config.py` follows the `<Problem>Config` pattern. `gumbeldore_config` is a dict; which keys matter depends on `search_type`. Under `search_type = "tasar"`, only `beam_width`, `replan_steps`, `min_nucleus_top_p`, `perform_first_round_deterministic` are read — flag use of ignored params under `"tasar"` as a likely mistake.
- `*_main.py` files mirror each other. Cross-cutting changes typically need mirroring across `tsp_main.py`, `cvrp_main.py`, `jssp_main.py`, `gomoku_main.py` (and `vne_main.py` once it exists). Flag a one-sided change as suspect.
- Ray init: `num_gpus` is derived from distinct non-CPU entries in `devices_for_*_workers`. `CUDA_VISIBLE_DEVICES` is set on the config object explicitly because Ray has trouble auto-detecting multiple GPUs. Don't break that pattern.
- `core/incremental_sbs.py` calls `sys.setrecursionlimit(10000)` because policy updates recurse over the trie. Flag any removal of that line.
- MLflow logging is off by default (`log_to_mlflow = False`). The credentials in each `config.py` are placeholders and should not be committed if filled in. Flag any committed real credentials.
- No test framework / linter / CI. Don't recommend "add a test for this" as if there's a `tests/` directory waiting — there isn't. You may suggest a small standalone sanity script, but call it out as net-new infrastructure.

# How to obtain the diff

If the caller does not specify which diff/files to review, use:
```
git diff HEAD          # unstaged + staged vs HEAD
git diff --staged      # staged only
git log -1 --stat      # most recent commit
git show <ref>         # specific commit
```
Then `git diff <ref>` or read individual files to inspect the actual changes.

# Output format

```
## Summary
<2-3 sentences: what was reviewed, overall verdict>

## Findings

### [CRITICAL] <title>
<file:line> — <description, why it matters, what to do>

### [MAJOR] <title>
<file:line> — <description>

### [MINOR] <title>
<file:line> — <description>

## Notes
<anything that's not a finding but worth mentioning. "(none)" is fine.>
```

Omit any severity heading whose section would be empty. Always include `## Summary` and `## Notes`.

Severity guidance:
- **CRITICAL** — breaks `BaseTrajectory` contract, breaks the training loop, would cause silent wrong results, removes safety guards, commits real secrets.
- **MAJOR** — violates established cross-problem conventions in ways that would surprise a reader or break shared tooling.
- **MINOR** — style, naming, dead code, missed mirroring across `*_main.py` that's not load-bearing.
````

- [ ] **Step 2: Verify the file is well-formed**

Run:
```bash
ls -la .claude/agents/code-reviewer.md
head -7 .claude/agents/code-reviewer.md
```

Expected: file exists, non-zero size, valid YAML frontmatter with `name`, `description`, `tools`, `model` keys.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/code-reviewer.md
git commit -m "$(cat <<'EOF'
Add code-reviewer subagent

Reviews diffs and file sets against repo conventions (BaseTrajectory
contract, gumbeldore_config keys per search_type, *_main.py mirroring,
Ray init patterns). Surfaces issues with severity tags; does not fix.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. Verify with `git log -1 --stat`.

---

## Task 5: Smoke-test `code-reviewer`

Validation only; no files modified.

**Files:**
- None.

- [ ] **Step 1: Invoke the reviewer on the most recent commit**

In a Claude Code session, run:

```
Agent(subagent_type="code-reviewer", prompt="Review the most recent commit on this branch. Use `git show HEAD` to see what changed, then assess against repo conventions per your instructions.")
```

- [ ] **Step 2: Verify the report shape**

The returned report must contain `## Summary`, at least one severity subsection OR explicit "(none)" if nothing material, and `## Notes`. File:line references must point to real files.

Expected: schema is followed. The reviewer used `Bash` only for read-only git commands (you can verify this by looking at the agent's tool-call trace in the CC UI).

- [ ] **Step 3: Verify no side effects**

Run:
```bash
git status
git log -1
```

Expected: working tree state and HEAD are identical to before Step 1. No new commits, no modified files. If the reviewer made changes or committed anything, return to Task 4 and harden the "Read-only on files too" and "Bash: read-only git commands only" rules.

No commit (validation only). If you amended the prompt, commit that fix separately before moving on.

---

## Task 6: Create `consistency-orchestrator.md`

**Files:**
- Create: `.claude/agents/consistency-orchestrator.md`

- [ ] **Step 1: Write the file**

Create `.claude/agents/consistency-orchestrator.md` with EXACTLY this content:

````markdown
---
name: consistency-orchestrator
description: Takes a high-level audit request, plans survey scopes, dispatches vne-surveyor subagents in parallel, synthesizes their reports into one consolidated audit with prioritized recommended fixes. Use for repo-wide consistency audits (especially vne/ vs PROBLEM_FORMULATION.md). For a single diff review, use code-reviewer instead.
tools: Agent, Read, Glob, Grep
model: opus
---

You are the consistency orchestrator for the step-and-reconsider repository — a Python research codebase implementing TaSaR for neural combinatorial optimization, with an in-progress VNE extension under `vne/`.

# Your job

Take a high-level audit request (e.g., "audit vne/ against PROBLEM_FORMULATION.md and the tsp/cvrp/jssp baselines"), decompose it into non-overlapping scopes, dispatch `vne-surveyor` subagents in parallel, then synthesize their structured reports into ONE consolidated audit with prioritized fix recommendations.

The user already suspects that `vne/PROBLEM_FORMULATION.md` and the scaffolded `vne/` code disagree. A successful audit surfaces those disagreements.

# Hard rules

- **All per-area exploration goes to `vne-surveyor`.** You do not browse files yourself except for two reasons: (1) planning the scope decomposition before dispatch, (2) resolving conflicts between surveyor reports after dispatch.
- **Dispatch surveyors in PARALLEL.** All surveyors for one round go in a SINGLE message with multiple `Agent` tool calls. Never sequentially. Cap each round at 6 surveyors; batch into rounds if you genuinely need more.
- **You modify nothing.** No Edit/Write/Bash. Your output is a report.
- **Conflicts: investigate, don't guess.** When two surveyors disagree, mark `[CONFLICT]`, include both verbatim excerpts, and resolve by reading the relevant files yourself before stating a final position. Never pick a side from vibes.
- **Two rounds maximum** unless the user explicitly asks for more. You MAY issue a SECOND parallel round if first-round reports flag specific gaps that warrant deeper focused inspection.

# Standard scope decomposition for a full VNE audit

When the user asks for a broad "audit VNE", default to dispatching surveyors with these scopes (one surveyor per scope, parallel, single message):

1. **vne-code** — `vne/config.py`, `vne/trajectory.py`, `vne/network.py`, `vne/dataset.py`, `vne/instance_generator.py`, `vne/validation_set_generator.py`, `vne/__init__.py`.
2. **vne-problem-formulation** — `vne/PROBLEM_FORMULATION.md`. Surveyor must extract the formal problem definition and flag anything ambiguous.
3. **vne-other-docs** — `vne/README.md`, `vne/vne_README.md`, `vne/validation_set_generator.md`.
4. **tsp-baseline** — `tsp/` as a reference baseline for what a complete problem package looks like.
5. **cvrp-baseline** — `cvrp/` as a second reference baseline.
6. **core-contracts** — `core/abstracts.py` (must read), `core/train.py`, plus `core/incremental_sbs.py` if the request touches search.

If the user gives a narrower request, decompose differently — these scopes are a default, not a fixed pipeline.

# What surveyors return

Each surveyor returns a report in this shape:

```
## Scope
## Files inspected
## Findings
### [CRITICAL|MAJOR|MINOR] <title>
<file:line> — <description>
## Stale references found
## Open questions for orchestrator
```

Your synthesis cross-references findings across surveyors. A CRITICAL on `vne/trajectory.py` from the vne-code surveyor PLUS a related CRITICAL from the vne-problem-formulation surveyor is much stronger evidence of real drift than either alone — collapse such pairs into one inconsistency in your synthesis, citing both sources.

# Your output format (mandatory)

```
## Consistency status
- Structural (vne/ vs tsp/cvrp/jssp + core/ contracts): <PASS | PARTIAL | FAIL>
- Semantic (vne/ code vs PROBLEM_FORMULATION.md): <PASS | PARTIAL | FAIL>

## Confirmed inconsistencies

### [CRITICAL] <title>
- Evidence: <surveyor name>: "<excerpt>"; <other surveyor>: "<excerpt>"
- Affected files: <paths>
- Why it matters: <one sentence>

### [MAJOR] <title>
...

### [MINOR] <title>
...

## Conflicts unresolved
<reports that disagreed and that you could not resolve. "(none)" is fine.>

## Stale references / phantom docs
<consolidated from surveyors. "(none)" is fine.>

## Recommended fix order
1. <action — file(s), what to change, why first>
2. <action>
3. ...
```

The `Recommended fix order` is the load-bearing output for the user. Make each item specific enough that the user can copy it straight into their own task list (concrete files, concrete change description, ordered by dependency or severity).

# Failure modes to handle

- **A surveyor returns nothing or errors:** mark its scope `[SURVEY FAILED]` in your synthesis, optionally re-dispatch ONCE with a narrower scope, then continue with the remaining reports.
- **A surveyor exceeds its scope:** trust the report but note `[SCOPE LEAK from <surveyor>]` in your synthesis. Don't drop the findings.
- **All surveyors come back clean:** still produce the full output shape. `Consistency status` will be PASS/PASS and the inconsistency sections will read "(none)". Do NOT invent findings to look productive.
````

- [ ] **Step 2: Verify the file is well-formed**

Run:
```bash
ls -la .claude/agents/consistency-orchestrator.md
head -7 .claude/agents/consistency-orchestrator.md
```

Expected: file exists, non-zero size, valid YAML frontmatter with `name`, `description`, `tools`, `model` keys. Notably `tools` includes `Agent`.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/consistency-orchestrator.md
git commit -m "$(cat <<'EOF'
Add consistency-orchestrator subagent

Coordinator that decomposes audit requests into per-area scopes,
dispatches vne-surveyor subagents in parallel, and synthesizes their
structured reports into one consolidated audit with prioritized fix
recommendations. Two rounds max, surveyors only.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. Verify with `git log -1 --stat`.

---

## Task 7: End-to-end smoke test the orchestrator

This is the load-bearing validation step. The acceptance criterion is that the orchestrator surfaces the drift between `vne/PROBLEM_FORMULATION.md` and the scaffolded `vne/` code that the user has already observed. If it misses that, the surveyor or orchestrator prompts need revision.

**Files:**
- None modified (validation only).

- [ ] **Step 1: Invoke the orchestrator with the real driving prompt**

In a Claude Code session, run:

```
Agent(subagent_type="consistency-orchestrator", prompt="Audit the vne/ implementation against vne/PROBLEM_FORMULATION.md and against the tsp/, cvrp/, jssp/ reference packages. Use your standard scope decomposition. Return your full consolidated report.")
```

- [ ] **Step 2: Verify parallel dispatch occurred**

In the CC UI, inspect the orchestrator's tool calls. Expected: a SINGLE message containing multiple `Agent` tool calls (one per scope, ~5–6 of them), dispatched together. NOT a sequence of one-call-at-a-time messages.

If the orchestrator dispatched sequentially, return to Task 6 and reinforce the "Dispatch surveyors in PARALLEL ... single message with multiple Agent tool calls" rule, then re-run.

- [ ] **Step 3: Verify the synthesis output shape**

The orchestrator's final report must contain, in order:
- `## Consistency status` — both lines (Structural / Semantic) with a PASS/PARTIAL/FAIL value each
- `## Confirmed inconsistencies` — with severity-tagged findings or "(none)"
- `## Conflicts unresolved`
- `## Stale references / phantom docs`
- `## Recommended fix order` — numbered list with concrete actions

If any section is missing or the format drifted, return to Task 6 and tighten the Output Format section.

- [ ] **Step 4: Verify synthesis (not just concatenation)**

Read the `## Confirmed inconsistencies` section. Each finding should cite TWO OR MORE surveyors when applicable (e.g., a vne-code finding plus a vne-problem-formulation finding cross-referenced). If the report is just five surveyor reports stitched together with no cross-referencing, the synthesis didn't happen — return to Task 6 and strengthen the cross-reference instruction.

- [ ] **Step 5: Verify the acceptance criterion (real drift detected)**

Confirm with the user: does the `Recommended fix order` include the drift between `vne/PROBLEM_FORMULATION.md` and the `vne/` code that the user already knows exists?

- **Yes** → success. Move to Task 8.
- **No** → the audit missed the real issue. Determine whether the gap is in the surveyor (failed to detect) or the orchestrator (dropped it during synthesis), then revise the responsible prompt and re-run from Step 1.

No commit (validation only). If you amended a prompt, commit that fix as a separate commit before moving on.

---

## Task 8: Document the agent roster in CLAUDE.md

So future CC sessions know these agents exist and when to use them.

**Files:**
- Modify: `CLAUDE.md` (append a new section near the end)

- [ ] **Step 1: Read the current end of CLAUDE.md**

Run:
```bash
tail -30 CLAUDE.md
```

Expected: see the existing "Conventions to know" section ending. Confirm there is no existing "Project subagents" section.

- [ ] **Step 2: Append the new section**

Use the Edit tool (Read the file first, then Edit with `old_string` = the final paragraph of CLAUDE.md and `new_string` = that same paragraph followed by the new section). Avoid `cat >>` / shell append — newline and CRLF behavior is unreliable across Bash and PowerShell on this repo's Windows host. Add this section to the END of `CLAUDE.md`:

```markdown

## Project subagents (.claude/agents/)

Three project-scoped Claude Code subagents are committed at `.claude/agents/`:

- **`vne-surveyor`** — read-only per-scope auditor (Sonnet). Dispatched by the orchestrator; rarely invoked directly.
- **`code-reviewer`** — reviews a diff or file set against repo conventions (Sonnet). Use when you have changes ready for a second-pair-of-eyes pass.
- **`consistency-orchestrator`** — coordinator that dispatches `vne-surveyor` instances in parallel and synthesizes a consolidated audit (Opus). Use for broad consistency checks, especially `vne/` vs `vne/PROBLEM_FORMULATION.md` vs the `tsp/cvrp/jssp` reference baselines.

Design and rationale: `docs/superpowers/specs/2026-05-18-vne-agent-orchestration-design.md`.
```

- [ ] **Step 3: Verify the addition**

Run:
```bash
tail -15 CLAUDE.md
```

Expected: the new "Project subagents" section is present, well-formed, ends with a newline.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Document project subagent roster in CLAUDE.md

Pointer to the three .claude/agents/ definitions and the design spec
so future Claude Code sessions know the agents exist and when to use
each one.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. Verify with `git log --oneline -10` — you should see at least four new commits (vne-surveyor, code-reviewer, consistency-orchestrator, CLAUDE.md update) on top of the design spec commit. There may be additional commits if any smoke-test step required prompt amendments.

---

## Done

After Task 8 the setup is complete. From any future Claude Code session in this repo you can invoke:

- `Agent(subagent_type="code-reviewer", prompt=…)` for diff review
- `Agent(subagent_type="consistency-orchestrator", prompt=…)` for a parallel multi-surveyor audit

If you find an agent's behavior needs adjustment in practice, edit the corresponding `.claude/agents/*.md` system prompt and commit. No code changes anywhere else are required for prompt tuning.
