# AgentTailor (experiment fork)

This repository is an experimental fork focused on **training / ablation** around the AgentTailor-style multi-agent graph (spatial + temporal structure, critics, staged training). Core network and experiment code live under `AgentTailor/` and `experiments/`.

---

## Repository layout

| Path | Role |
|------|------|
| `AgentTailor/ATNetwork/` | Actor, critics, nodes, replay / interaction |
| `AgentTailor/agents/`, `prompt/`, `llm/` | Agent roles, prompts, LLM adapters |
| `experiments/` | Entry scripts (`train4*.py`) and `train_base.py` |
| `dataset/` | Datasets and loaders (AQuA, GSM8K, MMLU, HumanEval, etc.) |
| `experiments_util/` | Judges, loaders, helpers for experiments |
| `.env.template` | Copy to `.env` and fill API keys / base URLs |

Generated runs, caches, and local secrets should stay out of git (see `.gitignore`).

---

## Quick start

**Before you run anything, pick the entry script for your domain and edit hyperparameters there.**

1. **Where to tune hyperparameters**  
   - Open the matching file under `experiments/`, for example `experiments/train4aqua.py`, `experiments/train4gms8k.py`, `experiments/train4humaneval.py`, `experiments/train4mmlu.py`, `experiments/train4multiarith.py`, or `experiments/train4svamp.py`.  
   - In each file, adjust the dict returned by **`_build_config()`** (learning rates, sample caps, `llm_name`, `agent_names`, `epn_dims`, `optimized_spatial` / `optimized_temporal`, domain flags, etc.).  
   - Fields may be `None`; those are replaced inside **`experiments/train_base.py`** in **`train_all()`** with built-in defaults. Change defaults globally there only if you intend to affect every domain script.

2. **Python** 3.10+ recommended.

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**  
   Copy `.env.template` to `.env` and set at least:

   - `OPENAI_API` (and `OPENAI_BASE_URL` if you use a proxy or Azure-compatible endpoint)  
   - or DeepSeek variables if you use that stack.

5. **Run an experiment** (from the repo root), e.g.:

   ```bash
   python experiments/train4aqua.py
   ```

   Other domains:

   - `python experiments/train4gms8k.py`
   - `python experiments/train4humaneval.py`
   - `python experiments/train4mmlu.py`
   - `python experiments/train4multiarith.py`
   - `python experiments/train4svamp.py`

Each script builds a config dict via `_build_config()` and calls `train_base.train_all(**config)` — use **keyword unpacking**, not a single dict positional argument. Optional: `_build_config(overrides={...})` where the script supports it.

---

## Data

- Some splits ship under `dataset/` (e.g. AQuA JSONL, GSM8K, HumanEval, MultiArith, Svamp).
- **MMLU** and other large corpora may be missing until you download them; use helpers such as `dataset/MMLU/download.py` and follow any README or comments in that folder.


