from typing import Any, Union

from molgen.rewards.functions.multi_reward import MultiReward
from molgen.rewards.functions.rdkit_rewards import PenalizedLogPReward, QEDReward, pIC50Reward
from molgen.rewards.reward import AbstractReward

name_to_reward: dict[str, type[AbstractReward]] = {
    "QED": QEDReward,
    "PlogP": PenalizedLogPReward,
    "pIC50": pIC50Reward     # Place holder for SSM-DTA integration
}


def get_rewards(rewards_dict: dict[str, Any]) -> Union[AbstractReward, list[AbstractReward]]:
    rewards: list[AbstractReward] = []
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
