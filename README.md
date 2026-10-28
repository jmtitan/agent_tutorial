# Agentic memory exercises

Hands-on notebooks that each rebuild one LLM-agent memory mechanism from a paper
as small, inspectable code. Every exercise has a contract, a hint, executable
checks, and a collapsed reference answer.

| Lesson | Paper | Topics | Needs |
| --- | --- | --- | --- |
| [01 · Generative Agents](notebooks/01_generative_agents.ipynb) | [Park et al., 2023](https://arxiv.org/abs/2304.03442) | memory stream, three-factor retrieval, reflection | Python stdlib only |
| [02 · MemGPT](notebooks/02_memgpt_qwen3.ipynb) | [Packer et al., 2023](https://arxiv.org/abs/2310.08560) | working vs. archival memory, function calls, heartbeat loop | Qwen3-4B on vLLM for the live task |
| [03 · A-MEM](notebooks/03_amem.ipynb) | [Xu et al., 2025](https://arxiv.org/abs/2502.12110) | note construction, linking, memory evolution | offline fixtures; vLLM optional |

## Usage

Open a notebook with a Python 3.10+ kernel and work top to bottom, replacing each
`raise NotImplementedError(...)` and running the check below it. Live cells in
lessons 2–3 expect an OpenAI-compatible vLLM server; setup steps are in each
notebook. These are teaching demos of the mechanisms, not reproductions of the
papers' results.

## `util.py`

Shared, stdlib-only helpers: `VLLMClient` (request budget, token-limit checks,
optional JSON-schema output, `requests.jsonl` logging), `new_run`, `save_json`,
`load_json`, and `decode_json`. Run artifacts must live under `OUTPUT_ROOT`
(edit it at the top of `util.py`); the client accepts only loopback URLs, so
use an SSH tunnel for a remote server.

```python
from util import OUTPUT_ROOT, VLLMClient, new_run

run_dir = new_run(OUTPUT_ROOT / "demo-001")
client = VLLMClient("http://127.0.0.1:8102", run_dir, model="Qwen/Qwen3-4B",
                    max_model_len=4096, max_tokens=512, request_limit=13)
client.verify_model()
print(client.complete([{"role": "user", "content": "Hello"}]))
```
