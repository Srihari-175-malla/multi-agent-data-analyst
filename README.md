# Multi-Agent AI Data Analyst

A real agentic system — not a single LLM call wrapped in a chat box. A user
uploads tabular data (CSV/Excel) and asks an analytical question like
*"Why did sales decrease last month?"*. A **Manager Agent** decomposes the
question and delegates to specialist agents that actually **execute SQL,
run Python/pandas analysis, and perform statistical tests**. A **Critic Agent**
validates the combined evidence before a **Report Agent** writes the final,
citation-grounded report.

## Architecture

```
                     User
                       |
                 Manager Agent  (plans, delegates, aggregates, retries on REVISE)
                       |
        +--------------+--------------+
        |              |              |
   SQL Agent      Python Agent   Statistics Agent
        |              |              |
     executes       executes       runs tests
   read-only SQL   sandboxed        (t-test, trend,
   (DuckDB)         pandas/plots     seasonality,
                     (matplotlib)     changepoint...)
        |              |              |
        +--------------+--------------+
                       |
                 Critic Agent   (checks grounding, stats validity, asks for
                       |         revision -> loops back to workers, bounded retries)
                       |
                 Report Agent  (writes final markdown report + embeds charts)
                       |
              Dashboard + Report
```

Every agent is a **real tool-calling loop** against Claude (`src/llm/llm_client.py`
+ `src/agents/base_agent.py`): the model is given tool schemas, actually invokes
them, and the tool results (SQL rows, DataFrame summaries, test statistics,
saved chart paths) are fed back until the agent produces a final answer.
Nothing here is templated or faked — SQL genuinely executes against DuckDB,
Python genuinely executes in a restricted sandbox, and statistical tests
genuinely run via `scipy`/`statsmodels`.

## What makes this "agentic" rather than an API wrapper

- **Tool calling**: each specialist agent has its own tool (`execute_sql`,
  `execute_python`, `run_statistical_test`) and loops turn-by-turn until done.
- **Agent memory**: a shared blackboard (`src/memory/agent_memory.py`) accumulates
  every agent's evidence so later agents (Critic, Report) can ground claims in it,
  plus a persisted long-term log of past sessions.
- **Retries with feedback**: if the Critic rejects the evidence, the Manager
  routes specific, targeted feedback back to the responsible agent and re-runs
  it — bounded by `max_revision_rounds` — rather than blindly retrying everything.
- **Guardrails**: SQL is statically checked to be read-only/single-statement
  before execution; Python is AST-checked against an import/builtin allowlist
  and run with a wall-clock timeout; the Critic checks that every numeric claim
  in a draft report traces back to something in the evidence blackboard.
- **Evaluation**: `src/evaluation/evaluator.py` runs a benchmark question set
  end-to-end and reports task success rate, critic-approval rounds, tool error
  rate, latency, and LLM-judged report faithfulness.

## Tech stack

- **LLM**: Anthropic Claude, native tool use (`anthropic` SDK)
- **SQL execution**: DuckDB (in-process, reads CSV/Excel directly, read-only guardrail)
- **Python sandbox**: restricted `exec()` with AST allowlisting + `multiprocessing` timeout
- **Stats**: `scipy`, `statsmodels`
- **Plotting**: `matplotlib`, saved as PNG artifacts referenced by the report
- **API**: FastAPI
- **UI**: Streamlit (upload data, ask questions, inspect full agent trace, view report)
- **Infra**: Docker + docker-compose

## Repository layout

```
src/llm/                Claude tool-use client
src/agents/              base tool-calling loop + Manager/SQL/Python/Statistics/Critic/Report agents
src/tools/               the actual executable tools (SQL, python exec, stats, plotting)
src/guardrails/          SQL/Python static safety checks + report grounding check
src/memory/              shared blackboard + persisted session log
src/orchestration/       the end-to-end workflow with the retry/revision loop
src/data/                CSV/Excel -> DuckDB loader
src/evaluation/          benchmark harness
api/                     FastAPI app (upload, analyze, fetch report)
frontend/app.py          Streamlit dashboard
scripts/                 CLI entry point for running an analysis from the terminal
tests/                   Pytest unit tests (guardrails, tools, memory, fusion logic)
```

## Quickstart

### 1. Configure
```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
```

### 2. Docker
```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- UI:  http://localhost:8501

### 3. Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000
streamlit run frontend/app.py   # separate terminal
```

### 4. CLI
```bash
python scripts/run_analysis.py --data data/sample/sales.csv \
  --question "Why did sales decrease last month?"
```

## Example (API)

```bash
curl -X POST localhost:8000/upload -F "file=@data/sample/sales.csv"
# -> {"dataset_id": "sales", "schema": {...}}

curl -X POST localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "sales", "question": "Why did sales decrease last month?"}'
```

The response includes the final report, referenced chart file paths, and the
**full audit trail**: every tool call each agent made, its arguments, its raw
result, and the Critic's verdict at each revision round.

## Evaluation

```bash
python -m src.evaluation.evaluator --benchmark data/eval/benchmark.jsonl
```

## License
MIT
