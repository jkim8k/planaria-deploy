# PLANARIA_ONE

Single-file agent runtime (`planaria_core.py`) with:
- workspace isolation (`workspace_<agent>`)
- tool-using chat loop (web/file/shell/memory/schedule/integrations)
- Telegram + CLI + IPC modes
- background scheduler
- multi-route LLM fallback

This repository is intentionally documentation-light.
`README.md` is the primary 운영 문서이며, 기타 보고서/백업 문서는 보조 참고 자료로 취급한다.

## 1. Project Scope

Core entrypoints:
- `planaria_core.py`: engine + tools + runners
- `planaria_run.sh`: launcher wrapper (`uv run python ...`)
- `tester_planaria_one.py`: E2E evaluator
- `tester_v4.py`: deployment-gate evaluator
- `tester_common.py`: shared tester utilities
- `test_*.py`: unit/integration tests

## 2. Local Setup

Requirements:
- Python 3.11+
- `uv`

Install deps:

```bash
uv sync
```

Initialize workspace:

```bash
uv run python planaria_core.py --agent fox --init-only
```

Run CLI:

```bash
./planaria_run.sh --agent fox --mode cli
```

One-shot run:

```bash
./planaria_run.sh --agent fox --message "오늘 주요 AI 뉴스 요약해줘"
```

## 3. Runtime Modes

- `--mode cli`: interactive terminal
- `--mode telegram`: Telegram long-polling
- `--mode all`: CLI + Telegram
- `--mode ipc`: JSONL stdin/stdout mode (programmatic)

Scheduler always runs in background for non-IPC modes.

### IPC protocol (current)

Input (JSONL):
- `{"type":"message","content":"...","source":"tester","user_id":"u1","turn_id":"optional"}`
- `{"type":"cancel","turn_id":"optional"}`
- `{"type":"exit"}`

Output (JSONL):
- `{"type":"ready", ...}`
- `{"type":"tool_call","name":"...","args":{...},"iteration":N,"turn_id":"..."}`
- `{"type":"reflection_check", ... ,"turn_id":"..."}`
- `{"type":"reflection_retry", ... ,"turn_id":"..."}`
- `{"type":"cancel_ack","turn_id":"..."}`
- `{"type":"response","content":"...","tool_calls":[...],"elapsed_sec":N,"cancelled":bool,"turn_id":"..."}`
- `{"type":"error","message":"...","turn_id":"..."?}`

Notes:
- `turn_id`는 요청-응답 상관관계를 위한 optional 필드이며, 전달되면 outbound 이벤트에도 에코된다.
- `cancel`은 cooperative cancel 신호로 처리된다.

## 4. Workspace Model

Each agent is isolated under `workspace_<agent>/`:

- `config.json`: runtime config
- `secrets.json`: fallback secret store (when OS keyring unavailable)
- `memory/memory.jsonl`: turn memory
- `memory/graph_memory.jsonl`: graph facts
- `data/scheduled_tasks.jsonl`: pending schedules
- `skills/`: external tools (manifest + runner)
- `logs/runtime_log.jsonl`: runtime events
- `logs/llm_trace.jsonl`: full LLM request/response traces

Memory graph policy:
- L1/L2/L3 facts are stored as graph triples (`entity`-`relation`-`target_entity`).
- Connectivity is enforced: if `target_entity` is omitted, the engine links the fact to a layer hub node.
- This keeps long-term traversal possible instead of isolated fact islands.

## 5. Scheduling (One-shot + Recurring)

Internal tool: `schedule_task`

Parameters:
- `run_at_utc` (required): ISO-8601 UTC timestamp
- `task_prompt` (required): prompt executed at run time
- `recurrence` (optional): `auto | none | daily | weekly | monthly`
- `interval` (optional): positive integer, default `1`

Behavior:
- `recurrence=none`: one-shot, removed after execution
- `recurrence=daily|weekly|monthly`: re-scheduled to next run after execution
- `recurrence=auto`: inferred from `task_prompt` keywords (`매일/매주/매월`)

## 6. Fallback Strategy

LLM client supports:
- primary route + fallback routes
- per-route cooldown (circuit breaker)
- last-route adaptive timeout
- persistent route state (`logs/llm_route_state.json`)

To reduce fallback dependence, the engine now handles these locally (no LLM call):
- short greeting/identity dialog
- local time questions

## 7. Test Commands

Fast checks:

```bash
python3 -m py_compile planaria_core.py tester_planaria_one.py
uv run pytest -q test_encoding_fix.py test_config_fix.py test_reflection.py
uv run pytest -q test_onboarding.py test_integrations.py test_skill_pipeline.py
```

E2E evaluator:

```bash
python3 tester_planaria_one.py --agent fox --task 1,31,34 --timeout 240
```

Tester architecture and branch audit:

```bash
cat TESTER_PROJECT_AUDIT_AND_GUIDE.md
```

## 8. Notes

- `workspace_*` may be dirty during local runs (memory/schedule/log changes).
- `call_external_tool.sh` is used for external skill execution.
- Keep this README aligned with real code behavior.
- `TESTER_PROJECT_AUDIT_AND_GUIDE.md`, `BRANCH_RECURRING_DETAILED.md`, `backup/*`, `*.bak.py`, `*.pre_refactor.py`는 히스토리/감사용 자료다.

## 9. Thinking Mechanism Upgrade Plan

This project is now evolving from plain tool-calling into a stronger reasoning loop with two goals:

1. Association (연관성)
- If user asks a factual question, the agent should not rely only on the first web query.
- It should also check relevant workspace memory wording and use that as additional search perspective.
- Example: for latest LLM model questions, memory clues like `Gemma`, `Qwen`, `DeepSeek` should trigger supplemental retrieval.

2. Skepticism (의심)
- After drafting any conclusion, the agent should run one mandatory self-critique pass:
  - "Why are major model families missing?"
  - "Why are sources concentrated in one region/domain?"
- If gaps are detected, perform corrective retrieval and revise the final answer before completion.

Implementation policy (phase 1):
- Rule-based gap checks first (deterministic, testable, no new API).
- Runtime logs must record when associative/skeptical follow-up search is triggered.
- Keep overhead bounded: at most one associative + one skeptical follow-up per task loop.
- Plus one global LLM skepticism pass on final answer (`skepticism_pass` runtime event).
 - Skepticism uses multi-persona prompts:
   - `coverage_auditor`
   - `evidence_prosecutor`
   - `bias_sentinel`
