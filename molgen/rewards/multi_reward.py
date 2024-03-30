from collections import OrderedDict
from typing import Dict, List, Union, Optional

from molgen.rewards.reward import AbstractReward


# TODO: Find a way to fix typing in this class
class MultiReward(AbstractReward):
    def __init__(self, rewards: List[AbstractReward], name: Optional[str]=None) -> None:
        super(MultiReward, self).__init__(name=name)
        self.rewards = rewards

    def __call__(self, smiles: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(smiles, str):
            smiles_reward = [reward_fn(smiles) for reward_fn in self.rewards]
            final_reward = sum(smiles_reward)  # type: ignore
            
            return final_reward
        else:
            smiles_rewards = {}
            for reward in self.rewards:
                smiles_rewards[str(reward)] = reward(smiles)

            # rewards is not a dictionary mapping from a name to list of reward values
            # the following parse the dictionary into a list of lists
            # in a way that the first sub-list in the list contains
            # all the different rewards for the first SMILES molecule and so forth
            if not self.eval:
                rewards = list(zip(*list(smiles_rewards.values())))
                rewards = [sum(rewards) for rewards in rewards]

                return rewards

            else:
                return smiles_rewards  # type: ignore


    @AbstractReward.eval.setter  # type: ignore
    def eval(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise ValueError("eval can only be set to boolean values")

        for reward in self.rewards:
            if hasattr(reward, "_eval"):
                reward.eval = val
        AbstractReward.eval.fset(self, val)

    def get_reward(self, name: str) -> AbstractReward:
        for reward in self.rewards:
            if str(reward) == name:
                return reward
        else:
            raise ValueError(f"{name} is not present in the reward list")

    def __str__(self) -> str:
        name = ""
        for reward in self.rewards:
            name = name + f"{str(reward)}_"

        return name
