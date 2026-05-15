from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to Python path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments import train_base

def _build_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "agent_names": ["MathSolver"] * 5,
        "llm_name": "BlueShirtChat",
        "decision_method": "FinalRefer",
        "optimized_spatial": True,
        "optimized_temporal": True,
        "domain": "aqua",
        "num_rounds": 2,
        "lr_actor": 0.04,
        "lr_critic": 2e-3,
        "sparsity_weight": 0.0,
        "stage1_sample_count": 10,
        "stage2_sample_count": 30,
        "stage2_virtual_steps": 4,
        "lambda2": 0.02,
        "lambda3": 0.04,
        "stage3_virtual_steps": 5,
        "stage3_prune_ratio": 0.25,
        "lock_threshold": 0.001,
        "temperature": 1.0,
        "epn_dropout": 0.1,
        "critic_weight_decay": 0.01,
        "epn_dims": [train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes(),
        "stage2_logit_path": train_base.DEFAULT_STAGE2_LOGITS,
        "max_training_samples": 40,
        "max_validation_samples": 129,
    }
    if overrides:
        config.update(overrides)
    return config
import asyncio
import random
import time
import numpy as np
import torch
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls

SEED = 888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


async def train_aqua_with_stage3_stats(**config: Any) -> Dict[str, Any]:
    max_train = config.pop("max_training_samples", None)
    max_eval = config.pop("max_validation_samples", None)
    config.pop("dataset_split", None)
    config.pop("validation_split", None)
    return await train_base.train_all(
        **config,
        max_train_split_samples=max_train,
        max_test_split_samples=max_eval,
    )


def main() -> None:

    config = _build_config()
    start_wall_clock = time.time()
    result = asyncio.run(train_aqua_with_stage3_stats(**config))

    print("\n" + "=" * 80)
    print("Token statistics")
    print("=" * 80)
    prompt_tokens = int(result.get("prompt_tokens", int(PromptTokens.instance().value)))
    completion_tokens = int(result.get("completion_tokens", int(CompletionTokens.instance().value)))
    total_tokens = int(result.get("total_tokens", prompt_tokens + completion_tokens))
    cost = float(result.get("estimated_cost_usd", float(Cost.instance().value)))
    api_calls = int(result.get("api_calls", int(ApiCalls.instance().value)))
    wall_clock_seconds = float(result.get("wall_clock_seconds", time.time() - start_wall_clock))
    print(f"Prompt Tokens: {prompt_tokens:,}")
    print(f"Completion Tokens: {completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"API Calls: {api_calls:,}")
    print(f"Wall-clock Time: {wall_clock_seconds:.2f}s")
    print(f"Cost: ${cost:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ AQuA training interrupted by user.")
