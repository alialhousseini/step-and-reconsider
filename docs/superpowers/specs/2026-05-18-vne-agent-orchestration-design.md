# VNE Agent Orchestration — Design

**Date:** 2026-05-18
**Status:** Approved design, pending implementation plan
**Scope:** Set up a project-scoped Claude Code subagent roster to support development of the VNE extension under `vne/`, with a focus on surfacing drift between the formal problem definition (`vne/PROBLEM_FORMULATION.md`) and the scaffolded code.

---

## 1. Goal

Provide dev-time assistance — not a runtime/algorithm component — for the in-progress VNE (Virtual Network Embedding) extension. Two concrete needs:

1. Standard code review of changes against repo conventions (BaseTrajectory contract, mirroring across `*_main.py`, gumbeldore_config keys, Ray patterns).
2. A way to audit the VNE implementation for consistency — both structural (vs `tsp/cvrp/jssp` reference packages) and semantic (vs `vne/PROBLEM_FORMULATION.md`). The user has already observed drift between the formulation doc and the scaffolded code; the audit must reliably surface it.

## 2. Non-goals

- Not a runtime/algorithm contribution. Agents do not participate in TaSaR search.
- Not unattended cloud orchestration. The Managed Agents API is out of scope for this design.
- Not automatic. Agents run on demand from the user's CC session; no git hooks, no scheduled jobs.
- Not authoritative. Agents produce reports and reviews; they do not auto-fix code.

## 3. Architecture

### 3.1 File layout

```
.claude/
├── agents/
│   ├── code-reviewer.md
│   ├── consistency-orchestrator.md
│   └── vne-surveyor.md
└── settings.local.json   (existing)
```

All three agent definitions are project-scoped (committed to git). Each is a single markdown file with YAML frontmatter (`name`, `description`, `tools`, `model`) and a system prompt body — the standard Claude Code subagent format. CC auto-discovers anything under `.claude/agents/` and exposes them via the `Agent` tool's `subagent_type` parameter.

### 3.2 Invocation

- `Agent(subagent_type="code-reviewer", prompt=…)` — direct review of a diff or file set.
- `Agent(subagent_type="consistency-orchestrator", prompt=…)` — entry point for an audit. This is the only agent the user typically invokes directly aside from `code-reviewer`.
- `Agent(subagent_type="vne-surveyor", prompt=…)` — dispatched by the orchestrator. The user *may* invoke it directly for a one-off scoped survey, but the normal pattern is orchestrator-driven.

### 3.3 Subagent dispatch model

Claude Code subagents can themselves call the `Agent` tool, which is what makes the orchestrator pattern possible. There is no shared state between subagent calls — each surveyor returns a single text report to the orchestrator, which synthesizes them.

## 4. Agent specifications

### 4.1 `code-reviewer`

| Field | Value |
| --- | --- |
| Model | `sonnet` |
| Tools | `Read, Glob, Grep, Bash` |
| Implicitly excluded | `Edit, Write, NotebookEdit, Agent` (not listed in tools frontmatter → not granted) |

**Role:** Review a diff or named file set against repo conventions. Surface issues; do not fix them.

**Bash usage rule (enforced in the system prompt, not by tool list):** read-only git commands only (`git diff`, `git log`, `git show`, `git status`). No `git commit`, no `git checkout`, no file-modifying shell commands. The tools list cannot restrict bash subcommands; the prompt must.

**System prompt encodes:**
- The `BaseTrajectory` interface (`init_batch_from_instance_list`, `log_probability_fn`, `transition_fn`, `to_max_evaluation_fn`, `num_actions`) and the rule that subclasses must honor it.
- The convention that `tsp_main.py`, `cvrp_main.py`, `jssp_main.py`, `gomoku_main.py` mirror each other — changes to one typically need mirroring across the others (and the new `vne_main.py` once it exists).
- Standard `gumbeldore_config` keys per `search_type` (which params are read by `"tasar"` vs `"wor"` vs `"gumbeldore"` vs `"theory_gumbeldore"` vs `"beam_search"`), and the trap that many params are silently ignored under `"tasar"`.
- Ray initialization patterns from `main_train_cycle` and `reproduce.py`, including the `CUDA_VISIBLE_DEVICES` workaround.
- The gotcha that `to_max_evaluation_fn` returns an objective to MAXIMIZE (e.g., negative tour length).
- No test framework, no linter: review focuses on conventions and correctness, not on running anything.

**Output format:** Findings tagged `[CRITICAL] [MAJOR] [MINOR]` with file:line references, followed by a short summary.

### 4.2 `vne-surveyor`

| Field | Value |
| --- | --- |
| Model | `sonnet` |
| Tools | `Read, Glob, Grep` |
| Implicitly excluded | `Bash, Edit, Write, NotebookEdit, Agent` (not listed in tools frontmatter → not granted). Lack of `Agent` is the structural guard against recursive fan-out. |

**Role:** Audit ONE assigned scope and return a structured report. Read-only. Cannot dispatch further subagents.

**System prompt encodes:**
- The two audit dimensions: (a) structural consistency of `vne/` vs `tsp/cvrp/jssp` reference packages and against the `core/` contracts, and (b) semantic conformance between `vne/` code and `vne/PROBLEM_FORMULATION.md`.
- A fixed report schema:
  ```
  ## Scope
  <what was assigned>

  ## Files inspected
  - <path>
  - <path>

  ## Findings
  ### [CRITICAL] <title>
  <file:line> — <description>
  ### [MAJOR] <title>
  …
  ### [MINOR] <title>
  …

  ## Stale references found
  <files/symbols mentioned in docs that do not exist in code, or vice versa>

  ## Open questions for orchestrator
  <things the surveyor would need to check outside its scope to resolve>
  ```
- A hard rule: stay within the assigned scope. If the surveyor needs context from another directory, it raises an `Open question for orchestrator` instead of going to look itself.

### 4.3 `consistency-orchestrator`

| Field | Value |
| --- | --- |
| Model | `opus` |
| Tools | `Agent, Read, Glob, Grep` |
| Implicitly excluded | `Edit, Write, NotebookEdit, Bash` (not listed in tools frontmatter → not granted) |

The orchestrator does not call `TaskCreate`/`TaskUpdate` itself — task-tool availability inside subagents is not reliable to depend on. Instead, the orchestrator's report ends with a numbered `Recommended fix order` that the user can paste into their own task list in the parent session.

**Role:** Take a high-level audit request, plan the survey scopes, dispatch surveyors in parallel, synthesize their reports into one consolidated audit with prioritized recommended actions.

**System prompt encodes:**
- Standard scope decomposition for a full audit:
  - `vne/` code
  - `vne/PROBLEM_FORMULATION.md` (read against the code surveyor's findings)
  - The other `vne/*.md` files (`README.md`, `vne_README.md`, `validation_set_generator.md`)
  - `tsp/` (reference baseline)
  - `cvrp/` (reference baseline)
  - `jssp/` (reference baseline)
  - `core/` contracts (`abstracts.py`, `train.py`, search files as needed)
- Parallel dispatch rule: all surveyors for one round go in a SINGLE message with multiple `Agent` tool calls — never sequential. Cap at 6 concurrent surveyors per round; if more are needed, batch into rounds.
- Hard rule: all per-directory exploration goes to `vne-surveyor`. The orchestrator only reads files itself when planning scopes or resolving conflicts between surveyor reports.
- Conflict resolution: when two surveyors disagree, mark `[CONFLICT]`, include both excerpts verbatim, and resolve by reading the relevant files itself before stating a final position. Never pick a side from vibes.
- Permission to issue a SECOND parallel round if reports surface specific gaps that warrant deeper inspection.
- Synthesis output schema:
  ```
  ## Consistency status
  - Structural (vne/ vs tsp/cvrp/jssp): <PASS / PARTIAL / FAIL>
  - Semantic (vne/ vs PROBLEM_FORMULATION.md): <PASS / PARTIAL / FAIL>

  ## Confirmed inconsistencies
  ### [CRITICAL] <title>
  - Evidence: <surveyor excerpts>
  - Affected files: <…>
  ### [MAJOR] …
  ### [MINOR] …

  ## Conflicts unresolved
  <if any>

  ## Stale references / phantom docs
  <consolidated from surveyors>

  ## Recommended fix order
  1. <action>
  2. <action>
  …
  ```

## 5. Data flow

```
You ──"audit vne/ vs PROBLEM_FORMULATION.md"──▶ consistency-orchestrator
                                                        │
                                          1. Plans scopes (5–6 typically)
                                                        │
                                          2. Parallel dispatch (one message,
                                             multiple Agent tool calls)
                                                        │
                ┌───────────────┬─────────┴────────┬────────────────┬─────────────────┐
                ▼               ▼                  ▼                ▼                 ▼
         vne-surveyor    vne-surveyor       vne-surveyor      vne-surveyor     vne-surveyor
         (vne/ code)     (PROBLEM_FORM.)    (tsp/ ref)        (cvrp/ ref)      (core/)
                │               │                  │                ▼                 │
                └───────────────┴──── structured reports ───────────┴─────────────────┘
                                                        │
                                          3. Synthesis: cross-reference,
                                             severity-order, prioritize fixes
                                                        │
                                                        ▼
                                          Consolidated audit report ──▶ You
```

The orchestrator does NOT fix anything. Its output is a report. The user (or `code-reviewer` on a follow-up diff, or a later coding session) acts on it.

## 6. Error handling and edge cases

| Case | Handling |
| --- | --- |
| Surveyor fails / returns no report | Orchestrator continues with the reports it has, marks the missing scope as `[SURVEY FAILED]`, optionally re-dispatches once with a narrower scope. Never silently drops a scope. |
| Conflicting findings across surveyors | Mark `[CONFLICT]`, include both excerpts verbatim, orchestrator reads the relevant files itself before stating a final position. |
| Scope leakage (surveyor wants to peek elsewhere) | Forbidden by surveyor prompt. Surveyor raises an `Open question for orchestrator` instead. |
| Stale references in docs (e.g. CLAUDE.md mentions `MATHEMATICAL_FORMULATION.md` and `TASAR_CONNECTION.md`, which don't exist) | Surveyors include a `Stale references found` section. Orchestrator consolidates. |
| Recursion / accidental fan-out | Surveyors have no `Agent` tool. Only the orchestrator dispatches. Hard structural guard. |
| Parallelism explosion | Cap of 6 concurrent surveyors per round, baked into the orchestrator prompt. |

## 7. Cost and runtime expectations

A full audit run is roughly: 5–6 parallel Sonnet surveyors + 1 Opus orchestrator synthesis. Comparable to a moderate CC turn — bounded but not free. Worth knowing per invocation; not worth instrumenting.

## 8. Validation (post-implementation smoke checks)

After the three agent files are created, run these three checks before declaring the setup done:

1. **Surveyor smoke** — invoke `vne-surveyor` directly with scope = `vne/` code. Confirm the report follows the documented schema. If the shape is wrong, tighten the surveyor prompt.
2. **Code-reviewer smoke** — invoke `code-reviewer` on a small known diff (e.g., `git show HEAD`). Confirm it returns severity-tagged findings and did NOT attempt to edit anything.
3. **Orchestrator end-to-end** — invoke `consistency-orchestrator` with the real driving prompt: *"audit vne/ against PROBLEM_FORMULATION.md and the tsp/cvrp/jssp baselines."* Acceptance criteria:
   - Dispatches surveyors in parallel (single message, multiple Agent calls).
   - Synthesizes rather than just concatenating reports.
   - Produces actionable, prioritized fixes.
   - **Surfaces the real drift between PROBLEM_FORMULATION.md and the scaffolded code that the user has already observed.** This is the load-bearing criterion: if the orchestrator misses what the user has already noticed, the prompts need revision.

Failures are addressed by iterating on the relevant agent's system prompt. No code changes required anywhere else in the repo.

## 9. Intentionally out of scope

- Git hooks / automatic invocation on commit. (Easily addable later.)
- Persistence / cross-run diffing of audit reports. (Easily addable later.)
- An `editor` or `fixer` agent that modifies code. Strict separation of audit and modification is preserved.
- Other specialized agents (`math-vs-code-checker`, `pattern-mirror`). Deferred until we see whether `vne-surveyor`'s system prompt grows unwieldy.
