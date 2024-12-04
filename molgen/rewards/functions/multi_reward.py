from functools import reduce

from molgen.rewards.reward import AbstractReward
from molgen.rewards.reward_utils import agg_to_op


# TODO: Find a way to fix typing in this class
class MultiReward(AbstractReward):
    def __init__(self, rewards: list[AbstractReward], agg_type: str="add", name: str | None=None) -> None:
        super().__init__(name=name, scale=None)
        self.rewards = rewards
        self.op = agg_to_op(agg_type)

    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        if isinstance(smiles, str):
            smiles_reward = [reward_fn(smiles) for reward_fn in self.rewards]
            final_reward = reduce(self.op, smiles_reward)  # type: ignore

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
                rewards = [reduce(self.op, rewards) for rewards in rewards]

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
        AbstractReward.eval.fset(self, val) # type: ignore

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

    def __repr__(self) -> str:
        name = ""
        for reward in self.rewards:
            name = name + f"{reward.__repr__()}_"

        return name
