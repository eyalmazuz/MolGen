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
    else:
        agg: str = "add" if "agg" not in rewards_dict else rewards_dict["agg"]
        return MultiReward(rewards, agg_type=agg)

