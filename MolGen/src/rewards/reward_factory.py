from typing import Any, Dict, List

from src.rewards.reward import AbstractReward
from src.rewards.rdkit_rewards import QEDReward, PenalizedLogpReward
from src.rewards.multi_reward import MultiReward


name_to_reward: Dict[str, AbstractReward] = {
        "QED": QEDReward,
        "Plogp": PenalizedLogpReward,
        }


def get_rewards(rewards_dict: Dict[str, Dict[str, Any]]) -> AbstractReward:
    rewards = []
    for name, reward_config in rewards_dict.items():
        reward = name_to_reward[name](**reward_config)
        rewards.append(reward)

    if len(rewards) == 1:
        return rewards[0]
    else:
        return MultiReward(rewards)

