from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path (allow running this file directly)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments import train_base


def _build_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "agent_names": ["MathSolver"] * 5,
        "llm_name": "gpt-4o",
        "decision_method": "FinalRefer",
        "optimized_spatial": True,
        "optimized_temporal": True,
        "domain": "svamp",
        "num_rounds": 2,
        "lr_actor": None,
        "lr_critic": None,
        "sparsity_weight": None,
        "stage1_sample_count": None,
        "stage2_sample_count": None,
        "stage2_virtual_steps": None,
        "lambda2": None,
        "lambda3": None,
        "stage3_virtual_steps": None,
        "stage3_prune_ratio": None,
        "lock_threshold": None,
        "temperature": None,
        "epn_dropout": None,
        "critic_weight_decay": None,
        "epn_dims": [train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes(),
        "dataset_split": "train",
        "max_training_samples": 40,
        "validation_split": "test",
    }
    if overrides:
        config.update(overrides)
    return config


import asyncio
import copy
import random
import time
import numpy as np
import torch
from dataset.svamp_dataset import SvampDataset
from experiments_util.soft_judge import Train4SoftJudge
from AgentTailor.ATNetwork.edge_judge import EdgeJudge
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls

SEED = 888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def main() -> None:

    config = _build_config()
    start_wall_clock = time.time()
    result = asyncio.run(train_svamp_with_stage3_stats(**config))

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


async def train_svamp_with_stage3_stats(**config: Any) -> Dict[str, Any]:

    dataset_split = config.pop("dataset_split", "train")
    validation_split = config.pop("validation_split", "test")
    max_training_samples = config.pop("max_training_samples", 40)
    max_validation_samples = config.pop("max_validation_samples", 200)


    original_init = SvampDataset.__init__

    def patched_init(
        self,
        split: str = "test",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if split == "test":
            train_ds = object.__new__(SvampDataset)
            train_kwargs = dict(kwargs)
            train_kwargs.setdefault("max_samples", max_training_samples)
            original_init(
                train_ds,
                split=dataset_split,
                *args,
                **train_kwargs,
            )


            val_ds = object.__new__(SvampDataset)
            val_kwargs = dict(kwargs)
            val_kwargs.setdefault("max_samples", max_validation_samples)
            original_init(
                val_ds,
                split=validation_split,
                *args,
                **val_kwargs,
            )


            self.records = list(train_ds.records) + list(val_ds.records)
            self._split = "train+test"
            print(
                f"✅ Patched SvampDataset: {len(train_ds.records)} train + "
                f"{len(val_ds.records)} val(test) = {len(self.records)} total samples"
            )
            return


        return original_init(self, split=split, *args, **kwargs)

    SvampDataset.__init__ = patched_init

    try:
        train_base.HumanEvalDataset = SvampDataset
        result = await train_base.train_all(**config)

        return result
    finally:

        SvampDataset.__init__ = original_init

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Svamp training interrupted by user.")

