from typing import Any

from molgen.rewards.functions.multi_reward import MultiReward
from molgen.rewards.functions.rdkit_rewards import PenalizedLogPReward, QEDReward
from molgen.rewards.reward import AbstractReward

name_to_reward: dict[str, type[AbstractReward]] = {
    "QED": QEDReward,
    "PlogP": PenalizedLogPReward,
}


def get_rewards(rewards_dict: dict[str, Any]) -> AbstractReward:
    rewards: list[AbstractReward] = []
    for reward_config in rewards_dict["functions"]:
        reward_type = reward_config.pop("type")
        reward = name_to_reward[reward_type](**reward_config)
        rewards.append(reward)

    if len(rewards) == 1:
        return rewards[0]
    else:
        agg: str = rewards_dict.get("agg", "add")
        return MultiReward(rewards, agg_type=agg)
