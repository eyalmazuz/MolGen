from typing import Any, Dict, List, Type

from molgen.rewards.reward import AbstractReward
from molgen.rewards.functions.rdkit_rewards import QEDReward, PenalizedLogPReward
from molgen.rewards.functions.multi_reward import MultiReward


name_to_reward: Dict[str, Type[AbstractReward]] = {
        "QED": QEDReward,
        "PlogP": PenalizedLogPReward,
        }


def get_rewards(rewards_dict: Dict[str, Any]) -> AbstractReward:
    rewards: List[AbstractReward] = []
    for reward_config in rewards_dict["functions"]:
        reward_type = reward_config.pop("type")
        reward = name_to_reward[reward_type](**reward_config)
        rewards.append(reward)

    if len(rewards) == 1:
        return rewards[0]
    elif rewards_dict.get("goal_conditioned", False):
        return rewards
    else:
        agg: str = rewards_dict.get("agg", "add")
        return MultiReward(rewards, agg_type=agg)

