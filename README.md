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
| `.env.template` | **API only**: copy to `.env` and set keys / base URLs for your LLM provider |

Generated runs, caches, and local secrets should stay out of git (see `.gitignore`).

---

## Quick start

**1. Training hyperparameters (required before any run)**  

- **All** optimization / training knobs (`lr_*`, stage sample counts, virtual steps, dropout, etc.) are set **only** in each script’s **`_build_config()`** (e.g. `experiments/train4aqua.py`).  
- **`train_base.train_all()` does not read training hyperparameters from environment variables.** If any required field is left as `None`, `train_all` raises a clear error telling you to set it in `_build_config()`.  
- Each `train4*.py` ships a small block **`_TRAIN_NUMERIC_EXAMPLE`** with **non–paper-tuned example numbers** for a cheap smoke run. **Replace them** with your own settings before serious experiments or publication-related runs.  
- Optional: `lambda3` (adaptive actor LR in real steps) may stay `None` if you do not use that path.  
- Architecture-only default: if `epn_dims` is omitted, `train_all` may fill a minimal shape from `epn_concat_input_dim()` / `epn_head_hidden_sizes()`; you can still override `epn_dims` in `_build_config()`.

**2. Python** 3.10+ recommended.

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. API credentials (`.env` — not training hyperparameters)**  

Copy `.env.template` to `.env` and set only what your LLM client needs, for example:

- `OPENAI_API` (and `OPENAI_BASE_URL` if you use a proxy or compatible gateway)  
- or `DEEPSEEK_*` variables if you use that stack  

Do **not** put learning rates, sample counts, or other training knobs in `.env`; put them in `_build_config()` as above.

**5. Run an experiment** (from the repo root), e.g.:

```bash
python experiments/train4aqua.py
```

Other entry points: `train4gms8k.py`, `train4humaneval.py`, `train4mmlu.py`, `train4multiarith.py`, `train4svamp.py`.

Each script builds a dict with `_build_config()` and calls `train_base.train_all(**...)` — use **keyword unpacking** from that dict (plus any extra kwargs the wrapper adds, e.g. dataset size caps).

### `_build_config()` keys you should understand

| Area | Typical keys |
|------|----------------|
| Model / graph | `agent_names`, `llm_name`, `decision_method`, `optimized_spatial`, `optimized_temporal`, `epn_dims`, `num_rounds` |
| Optimizers / loss | `lr_actor`, `lr_critic`, `sparsity_weight`, `critic_weight_decay`, `lambda2`, `lambda3` |
| Stages | `stage1_sample_count`, `stage2_sample_count`, `stage2_virtual_steps`, `stage3_virtual_steps`, `stage3_prune_ratio` |
| Critics / EPN | `lock_threshold`, `temperature`, `epn_dropout` |
| Data caps (where used) | `max_training_samples`, `max_validation_samples`; GSM8K also `gsm8k_shuffle_seed` |

Domain-specific wrappers may `pop()` extra keys (e.g. `dataset_split`) before calling `train_all` so only supported arguments are passed.

---

## Data

- Some splits ship under `dataset/` (e.g. AQuA JSONL, GSM8K, HumanEval, MultiArith, Svamp).
- **MMLU** and other large corpora may be missing until you download them; use helpers such as `dataset/MMLU/download.py` and follow any README or comments in that folder.
