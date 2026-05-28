# AgentTailor (experiment fork)

Experimental fork for **multi-agent graphs** (spatial + temporal edges), **Critics (EPN)**, and **staged training**. Core code lives under `AgentTailor/` and `experiments/`.

After you complete the steps below, run training from the **repository root**:

```bash
python experiments/train4gms8k.py
```
---

## Repository layout

| Path | Role |
|------|------|
| `AgentTailor/ATNetwork/` | Actor, Critics, nodes, edge judges |
| `AgentTailor/agents/`, `prompt/`, `llm/` | Agent roles, prompts, LLM adapters |
| `experiments/` | Entry scripts (`train4*.py`) and `train_base.py` |
| `dataset/` | Benchmark data and loaders |
| `experiments_util/` | Judges and experiment helpers |
| `.env.template` |  — copy to `.env` |

---

## Quick start (executable checklist)

### 1. Environment

- **Python** 3.10+ (3.10 or 3.11 recommended).
- Run all commands from the **repository root** (entry scripts add the project root to `sys.path`).

```bash
pip install -r requirements.txt
```

The first run downloads `sentence-transformers` encoder weights; you need network access or a configured mirror.

### 2. API credentials (`.env`)

Copy the template and fill in keys for your backend. 

```bash
# Linux / macOS
cp .env.template .env

# Windows (PowerShell)
Copy-Item .env.template .env
```

| Backend | Variables | `_build_config()` |
|---------|-----------|-------------------|
| OpenAI-compatible | `OPENAI_API_KEY` or `OPENAI_API`; `OPENAI_BASE_URL` if using a proxy/gateway | `llm_name`: any model id string (e.g. `"gpt-4o"`) → `GPTChat` |
| DeepSeek | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | `llm_name`: `"DeepSeekChat"` |

Resolution logic is in `AgentTailor/llm/llm_registry.py`.

### 3. Data files

Confirm files exist **before** running the matching entry script:

| `domain` | Entry script | Required paths (from repo root) |
|----------|--------------|--------------------------------|
| `gsm8k` | `train4gms8k.py` | `dataset/gsm8k/gsm8k.jsonl` (included) |
| `aqua` | `train4aqua.py` | `dataset/aqua/dev.jsonl`, `dataset/aqua/test.jsonl` (included) |
| `svamp` | `train4svamp.py` | `dataset/Svamp/train.json`, `dataset/Svamp/test.json` (included) |
| `multiarith` | `train4multiarith.py` | `dataset/MultiArith/train.json`, `dataset/MultiArith/test.json` (included) |
| `humaneval` | `train4humaneval.py` | `dataset/humaneval/humaneval-py.jsonl`|
| `mmlu` | `train4mmlu.py` | Run `dataset/MMLU/download.py` first |

**Suggested first run:** `train4gms8k.py` or `train4svamp.py` (data already in the repo).

### 4. Hyperparameters (required)

Each `experiments/train4*.py` ships with training hyperparameters set to `None`. 
Open the entry script you plan to run and replace every `None` training field in **`_build_config()`** with explicit values. Field names and roles are listed in [Hyperparameter reference](#hyperparameter-reference); match `domain` and `agent_names` to the [entry script table](#entry-scripts-and-domain).

Constraints:

- `stage1_sample_count + stage2_sample_count` ≤ training split size (after any `max_training_samples` cap).
- `stage3_virtual_steps == 0` skips Stage3 inside `train_all` (no extra LLM eval loop there).
- `lambda3` may stay `None` (optional adaptive actor LR during real steps).

### 5. Run

```bash
python experiments/train4gms8k.py
```

Other entries: `train4aqua.py`, `train4humaneval.py`, `train4mmlu.py`, `train4multiarith.py`, `train4svamp.py`.

---

## Entry scripts and `domain`

`train_all()` selects datasets and edge judges from `domain`. It **must** match the script you run.

| Script | Set `"domain"` to | Typical `agent_names` |
|--------|-------------------|------------------------|
| `train4humaneval.py` | `humaneval` | `["CodeWriting"] * 5` |
| `train4aqua.py` | **`aqua`** | `MathSolver` + `AnalyzeAgent`, etc. |
| `train4gms8k.py` | `gsm8k` | `MathSolver` + `AnalyzeAgent`, etc. |
| `train4svamp.py` | `svamp` | `["MathSolver"] * 5` |
| `train4multiarith.py` | `multiarith` | math agents |
| `train4mmlu.py` | `mmlu` | `["AnalyzeAgent"] * 5`, etc. |


Registered agent names (must appear in `AgentRegistry`): `MathSolver`, `AnalyzeAgent`, `CodeWriting`, `AdverarialAgent`, `FinalRefer`, `FinalWriteCode`, `FinalDirect`, `FinalMajorVote`.

---


`train4gms8k.py` additionally accepts `gsm8k_shuffle_seed` (default `888` in the template) for shuffling `gsm8k.jsonl` before train/val/test slices.

---

## Training pipeline (in memory)

1. **Stage 1** — `real_execution` on the first `stage1_sample_count` training records; trains Actor and Critic.
2. **Stage 2** — on the next `stage2_sample_count` records: `virtual_execution` × `stage2_virtual_steps`, then `real_execution` per record.
3. **After Stage 2** — `critics.lock_critic()` freezes the EPN snapshot for virtual steps and Stage3.
4. **Stage 3** (only if `stage3_virtual_steps != 0`) — `copy.deepcopy(actor)` and `copy.deepcopy(critics)`; Stage2 spatial/temporal logits are copied into `stage3_actor`; evaluation runs on the test split inside `train_all`.

Metrics are returned from `train_all` and printed by each entry script.

---

## Hyperparameter reference

**Location:** only `experiments/train4<domain>.py` → `_build_config()` (optional `overrides` dict).

**Rule:** `train_all()` does **not** read training hyperparameters from environment variables. Listed fields must be explicit numbers, not `None`.

| Key | Role |
|-----|------|
| `agent_names` | Agent class per node; length = node count |
| `llm_name` | Model id or `DeepSeekChat` |
| `decision_method` | Aggregation node, e.g. `FinalRefer` |
| `domain` | Dataset keyword (see table above) |
| `optimized_spatial` / `optimized_temporal` | Train spatial / temporal edge logits |
| `num_rounds` | Actor dialogue rounds per sample |
| `lr_actor`, `lr_critic` | Adam learning rates |
| `sparsity_weight` | Sparsity term in policy loss |
| `stage1_sample_count`, `stage2_sample_count` | Training records per stage |
| `stage2_virtual_steps` | Virtual gradient steps per Stage2 record |
| `lambda2` | Stage2 virtual-step LR scaling (`virtual_execution`) |
| `lambda3` | Optional; scales actor LR with running accuracy when set |
| `stage3_virtual_steps` | `0` = skip Stage3 in `train_all`; `> 0` = run Stage3 eval loop |
| `stage3_prune_ratio` | Weak-edge prune ratio before Stage3 |
| `lock_threshold`, `temperature`, `epn_dropout` | Critic lock threshold, EPN tanh scale, dropout |
| `critic_weight_decay` | Adam `weight_decay` on critic parameters |
| `epn_dims` | EPN MLP shape (see below) |
| `gsm8k_shuffle_seed` | `train4gms8k.py` only — shuffle seed for `gsm8k.jsonl` |

---

## Network structure

### Actor graph

| Key | Effect |
|-----|--------|
| `agent_names` | Which agent classes instantiate each node |
| `optimized_spatial` | Learnable spatial edge logits |
| `optimized_temporal` | Learnable temporal edge logits |
| `num_rounds` | More rounds → deeper temporal messaging (higher API cost) |

See `AgentTailor/ATNetwork/Actor.py`, `AgentTailor/agents/`, `AgentTailor/prompt/`.

### Critic (EPN) — `epn_dims`

`Critics` expects `epn_dims = [input_dim, hidden…, 1]` where `input_dim` matches the concatenated encoder width.

Helpers in `experiments/train_base.py`:

- `epn_concat_input_dim(n_segments=5, embed_dim=384)` — default 5 segments × 384 (MiniLM-L6-v2 style).
- `epn_head_hidden_sizes()` — default `[1]`.

In `_build_config()`, set `epn_dims` to `[train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes()` unless you customize the head. If `epn_dims` is omitted, `train_all` fills the same default. Update `epn_concat_input_dim(...)` if you change encoder output size or segment count.

### Critic behavior knobs

`lock_threshold`, `temperature` (EPN output scaling), and `epn_dropout` are passed into `Critics(...)`. Tune them together with `lr_critic` and `critic_weight_decay`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `train_all: the following hyperparameters are None: ...` | Set every listed field in `_build_config()` for that script |
| `Stage1 + Stage2 samples exceed available train split size` | Lower `stage1_sample_count` / `stage2_sample_count` or raise `max_training_samples` |
| `Dataset file not found` / `GSM8K data not found` | Add data per [Data files](#3-data-files) or run MMLU download |
| `unexpected keyword argument 'max_training_samples'` | On `train4humaneval.py`, use `max_train_split_samples` / `max_test_split_samples` instead |
| Wrong benchmark loaded (e.g. AQuA runs GSM8K logic) | Set `"domain": "aqua"` in `train4aqua.py` |
| OpenAI 401 / connection errors | Check `.env` keys and `OPENAI_BASE_URL` |
| Very slow run or high cost | Lower `num_rounds`, sample counts, set `stage3_virtual_steps=0`, reduce `max_*_samples` |

---

## Data notes

- AQuA, GSM8K, Svamp, and MultiArith files are included under `dataset/`.
- **HumanEval:** place `humaneval-py.jsonl` at `dataset/humaneval/humaneval-py.jsonl`.
- **MMLU:** download via `dataset/MMLU/download.py` and follow comments in that folder before `train4mmlu.py`.

