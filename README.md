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

**1. Fill hyperparameters and (optionally) network fields in `_build_config()`**  

Open the domain script under `experiments/` (e.g. `train4aqua.py`) and edit the dict returned by **`_build_config()`**. See [Filling in hyperparameters](#filling-in-hyperparameters) and [Adjusting network structure](#adjusting-network-structure) below.

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

Each script builds a dict with `_build_config()` and calls `train_base.train_all(**...)` — use **keyword unpacking** (wrappers may `pop()` extra keys such as dataset caps before calling `train_all`).

---

## Filling in hyperparameters

**Where:** only inside `experiments/train4<domain>.py` → **`_build_config()`** (plus optional `overrides` if you use that pattern).

**Rule:** `train_base.train_all()` **does not** read training hyperparameters from environment variables. Required numeric fields must be **explicit numbers** (not `None`), or `train_all` will raise `ValueError` listing what is missing.

**Typical workflow**

1. Pick the entry script for your benchmark (`train4aqua.py`, …).  
2. Copy an existing `_build_config()` from another script if you want a template, then change domain-specific keys (`domain`, `llm_name`, caps).  
3. Ensure **every** row in the table below that applies to your run is set (same names as `train_all` keyword arguments).  
4. Keep **`stage1_sample_count + stage2_sample_count` ≤ size of the training split** after caps (otherwise `train_all` raises).  
5. Set **`lambda3`** only if you want adaptive actor LR during real steps; otherwise use `None`.

| Key | Role |
|-----|------|
| `agent_names` | List of agent role names, length = number of nodes (e.g. five `MathSolver` for math scripts). Must match entries usable by `Actor`. |
| `llm_name` | Backend model id / registry name your `llm` layer resolves (e.g. OpenAI or custom chat class). |
| `decision_method` | How the graph aggregates multi-agent output (e.g. `FinalRefer`). |
| `num_rounds` | Multi-round dialogue depth inside the actor per sample. |
| `lr_actor`, `lr_critic` | Adam learning rates for actor logits vs critic (EPN). |
| `sparsity_weight` | Weight for sparsity / structure regularization in policy loss. |
| `stage1_sample_count`, `stage2_sample_count` | How many training-split records are used in Stage 1 vs Stage 2. |
| `stage2_virtual_steps` | Virtual gradient steps **per** Stage-2 record. |
| `lambda2` | Mixing / scaling for Stage-2 virtual loss (see `virtual_execution` in `train_base.py`). |
| `lambda3` | Optional: scales actor LR with running accuracy when non-`None`. |
| `stage3_virtual_steps` | Set to `0` to skip Stage 3 evaluation loop entirely. |
| `stage3_prune_ratio` | Fraction of weak edges pruned before Stage 3 (when enabled in pipeline). |
| `lock_threshold`, `temperature`, `epn_dropout` | Critic (EPN) locking sharpness, softmax temperature, dropout. |
| `critic_weight_decay` | Adam `weight_decay` on critic parameters. |
| `epn_dims` | MLP shape for the critic head — see [Adjusting network structure](#adjusting-network-structure). |
| `stage2_logit_path` | Where Stage-2 actor logits (and optional critic file) are saved; default via `train_base.DEFAULT_STAGE2_LOGITS`. |
| `max_training_samples`, `max_validation_samples` | Optional caps on train / eval splits (wrappers pass these as `max_train_split_samples` / `max_test_split_samples`). |
| `gsm8k_shuffle_seed` | (`train4gms8k.py` only) RNG seed for shuffling the single JSONL before train/val/test slices. |

---

## Adjusting network structure

Most graph / architecture choices are **also** in `_build_config()` because they are passed straight into `train_all` → `Actor` / `Critics`.

### Multi-agent graph (Actor)

| Key | Effect |
|-----|--------|
| `agent_names` | Which agent classes are instantiated at each node. Changing length changes **node count**; keep consistent with prompts and edge templates in `AgentTailor`. |
| `optimized_spatial` | If `True`, actor maintains **trainable spatial edge logits / masks** among agents. |
| `optimized_temporal` | If `True`, same for **temporal** (round-to-round) edges. Turning one off fixes that part of the graph to a simpler pattern (see `Actor` construction). |
| `num_rounds` | More rounds = deeper temporal messaging per task (more LLM cost). |

For behavior beyond names (tool use, prompts), inspect `AgentTailor/ATNetwork/Actor.py`, `AgentTailor/agents/`, and `AgentTailor/prompt/`.

### Critic (EPN) MLP width — `epn_dims`

`Critics` in `AgentTailor/ATNetwork/Critics.py` expects `epn_dims` to be a list `[input_dim, hidden…, 1]`: the **first** element must match the **concatenated encoder width** fed into the critic (default setup: one vector per edge segment).

Helpers in `experiments/train_base.py`:

- **`epn_concat_input_dim(n_segments=5, embed_dim=384)`** — default assumes **5 segments × 384 dims** (MiniLM-L6-v2–style encoder).  
- **`epn_head_hidden_sizes()`** — returns **`[1]`**, i.e. one hidden width of `1` before the scalar head (you may replace with e.g. `[64, 1]` if you change `Critics` to match).

**Recommended pattern in `_build_config()`:**

```python
"epn_dims": [train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes(),
```

If you change the encoder output size or segment count in `Actor` / `Encoder`, update **`n_segments` / `embed_dim`** in `epn_concat_input_dim(...)` so the first dimension of `epn_dims` still matches the actual concatenated feature length; otherwise forward passes will error.

If you omit `epn_dims`, `train_all` fills **`[epn_concat_input_dim()] + epn_head_hidden_sizes()`** automatically — override in `_build_config()` whenever you want a custom head.

### Critic behavior knobs

`lock_threshold`, `temperature`, and `epn_dropout` are passed into **`Critics(...)`** — they control how sharply edges lock, softmax sharpness in edge scoring, and dropout on the EPN. Tune them in `_build_config()` together with `lr_critic` and `critic_weight_decay`.

### Deeper structural edits

If you need new edge types, new encoders, or different tensor shapes, you will edit **`AgentTailor/ATNetwork/`** (especially `Actor.py`, `Critics.py`, `Node.py`) and keep `_build_config()` in sync with any new constructor arguments (you may need to extend `train_all`’s signature and the dict in `train4*.py`).

---

## Data

- Some splits ship under `dataset/` (e.g. AQuA JSONL, GSM8K, HumanEval, MultiArith, Svamp).
- **MMLU** and other large corpora may be missing until you download them; use helpers such as `dataset/MMLU/download.py` and follow any README or comments in that folder.
