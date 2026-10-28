# Agentic memory exercises

Start with [01 · Generative Agents](notebooks/01_generative_agents.ipynb).
Then work through [02 · MemGPT with Qwen3-4B](notebooks/02_memgpt_qwen3.ipynb).
Continue with [03 · A-MEM: evolving linked notes](notebooks/03_amem.ipynb).
Each lesson covers one paper through a small, inspectable memory mechanism.
This first notebook covers Park et al. (2023), an early foundational LLM-agent
memory architecture, without claiming it was the first memory system.

## Lesson 2: real model, explicit memory operations

The second lesson explains MemGPT's visible working memory, external archival
memory, function executor, and heartbeat control flow. Four exercises implement
prompt visibility, paginated retrieval, validated memory operations, and a bounded
agent loop. The final task uses the real local Qwen3-4B model through vLLM to store
one synthetic project fact, start a fresh session, retrieve the fact, and answer.

The notebook contains the theory, step-by-step service instructions, offline
checks, live task, trace-reading questions, and collapsed reference answers.
`util.py` provides shared HTTP, scratch-only logging and JSON storage.
The notebook contains its action schema, response validation, and session
persistence helpers. The launcher contains its preflight and execution steps. `scripts/serve_memgpt.sbatch` uses the
existing uv-managed `env/llm`, local model weights, and one A100 for at most 20
minutes. It verifies inputs and resources, starts vLLM, runs a real smoke request,
then executes the reference notebook. By default the service stops afterward.
Set `KEEP_SERVING=1` to keep a verified service available for interactive use
until the job limit; endpoint and SSH-tunnel instructions are in the notebook.

The single live task has a limit of 13 generation requests including the smoke
check. Requests, responses, state, tool traces, and the pass/fail verdict are
stored under `outputs/dynamic_mem/tutorial/lesson02/` on scratch. A failure is
preserved as a failure; there is no simulated fallback or automatic budget
increase. This is a mechanism demonstration, not reproduction of MemGPT scores.

Offline validation for lesson 2: run `python -m pytest tests -p no:cacheprovider`
from this directory with a scratch `--basetemp` and `PYTHONDONTWRITEBYTECODE=1`.
No GPU is needed for those checks.

## Lesson 3: A-MEM and evolving linked notes

[Lesson 3](notebooks/03_amem.ipynb) covers the
[A-MEM paper, v11](https://arxiv.org/html/2502.12110v11), Sections 3.1–3.4.
Six exercises implement note construction, candidate retrieval, model-selected
links, old-note evolution, transactional insertion, and bounded retrieval.
Each includes a contract, hint, executable checks, and a collapsed reference
answer. The focus is preserving original evidence while updating derived
metadata and refreshing its search representation.

One workshop-relocation scenario distinguishes a relevant earlier event from
an unrelated office event. The offline version uses clearly labelled fixture
responses. Optional live cells reuse `util.py` and an existing Qwen3-4B vLLM
endpoint; set `AMEM_BASE_URL` to its loopback URL or SSH tunnel. The notebook
does not start a service or submit a GPU job. Stop before the live smoke cell
for offline-only study; running all cells includes real model requests.

The live scenario allows at most 10 generation requests, including its initial
note-construction smoke. It records effective settings, before/after memory,
insertion decisions, retrieved context, raw requests, and a pass/fail verdict
under `outputs/dynamic_mem/tutorial/lesson03/` on scratch. It never falls back
to fixture answers. Word-count embeddings, directed links, evolution limited to
linked candidates, and bounded one-hop retrieval are explicit teaching choices.
This demonstrates the mechanism, not the paper's benchmark scores.

Run `python -m pytest tests/test_amem_lesson.py -p no:cacheprovider` with
`PYTHONDONTWRITEBYTECODE=1` and a scratch `--basetemp` for offline verification.

## Shared utilities for future agents

Import `VLLMClient`, `new_run`, `save_json`, and `load_json` from `util.py`.
The shared module does not assume MemGPT tool names, heartbeat control flow,
or a particular memory structure. It uses only the Python standard library.

```python
from util import VLLMClient, new_run, save_json

run_dir = new_run(
    "/scratch/mingyan_root/mingyan/jmtang/outputs/dynamic_mem/tutorial/next_agent/run-001"
)
client = VLLMClient(
    "http://127.0.0.1:8102", run_dir,
    model="Qwen/Qwen3-4B", max_model_len=4096,
    max_tokens=512, request_limit=13,
    chat_template_kwargs={"enable_thinking": False},
)
client.verify_model()
answer = client.complete([{"role": "user", "content": "Hello"}])
save_json({"answer": answer, "agent_state": {}}, run_dir / "state.json")
```

The URL is the server root without `/v1`; use a local SSH tunnel for remote
services. Choose a fresh run directory each time and match `max_model_len` to
the actual server configuration. The client returns text by default. For a
different agent protocol, pass its JSON `schema` and use
`response_decoder=decode_json` (also imported from `util`), or supply a callable
that validates the raw completion response. Structured generation uses the
installed vLLM 0.8.5 `guided_json` API. Schema-constrained output still needs
semantic validation by the agent.

Temperature, seed, request timeout, chat-template arguments, token limits, and
request budget are constructor options. No tool schema or Qwen-specific thinking
setting is imposed by default. Token overflow fails before generation; attempted
generation consumes the request budget even if the response fails validation.
`requests.jsonl` stores the request, raw response, decoded `result`, or error.

`save_json` refuses to overwrite files; `load_json` restores the data exactly,
including any queue fields. The MemGPT notebook's local `save_memory` /
`load_memory` helpers exclude the transient queue and reset it for a new session.
New lessons should import the shared client directly and specify their own
settings, as the updated lesson 2 notebooks now do.

## Working through lesson 1

Open the notebook in a notebook-capable editor and select a Python 3.10+
kernel from an existing uv-managed environment. The lesson code uses only the
standard library; it needs no API keys, downloads, GPU, or project installation.
Work from top to bottom, replacing each `raise NotImplementedError(...)`.
Run the check directly below each function before continuing. Unfinished
exercises intentionally stop execution.

The seven exercises cover writing memories, cosine relevance, normalization,
three-factor retrieval scores, retrieval with access-time updates, cited
reflections, and memory-conditioned actions. Each includes a contract, hint,
and executable checks. Collapsed reference answers at the bottom are Markdown;
they never overwrite your work when you run the notebook.

The final three-observation episode checks the complete memory loop. Word-count
vectors, supplied importance scores, and scripted generation callbacks are
explicit teaching substitutions. Passing the checks demonstrates implementation
of the toy mechanism, not reproduction of the paper's experiments or LLM quality.

Restart the kernel and run all cells after finishing. Keep notebook execution
outputs cleared in the repository. Save executed copies and verification logs
under `/scratch/mingyan_root/mingyan/jmtang/outputs/dynamic_mem/tutorial/`.
The notebook itself performs no filesystem writes.

## Maintainer check

`tests/test_memory_tutorial.py` executes the notebook with its reference answers
in memory, checks the final memory state and actions, and verifies that leaving
each exercise unfinished is detected. It does not modify the learner notebook.
Run it with `python -m pytest tests/test_memory_tutorial.py -p no:cacheprovider`,
using a scratch `--basetemp` directory and `PYTHONDONTWRITEBYTECODE=1`.

Sources: [paper, Sections 4.1–4.3](https://arxiv.org/html/2304.03442v2),
[authors' code](https://github.com/joonspk-research/generative_agents).
