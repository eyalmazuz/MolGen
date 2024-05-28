from molgen.rewards.reward_factory import get_rewards
from molgen.rewards.functions.rdkit_rewards import QEDReward, PenalizedLogPReward
from molgen.rewards.functions.multi_reward import MultiReward


def test_get_single_reward_empty_dict():
    rewards_dict = {"functions":[
        {"type": "QED"},
        ]
    }
    
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, QEDReward)

    qed = reward("CCC")
    assert reward.scale is None

def test_get_single_reward_scale_in_dict():
    rewards_dict = {"functions":[
        {"type": "QED", "scale": "mult"},
        ]
    }
    
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, QEDReward)
    assert reward.scale is not None


def test_get_multiple_rewards_empty_list():
    rewards_dict = {"agg": "mul", "functions":[
        {"type": "QED",},
        {"type": "PlogP"},
        ]
    }
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, MultiReward)

    assert len(reward.rewards) == 2
    assert isinstance(reward.rewards[0], QEDReward)
    assert isinstance(reward.rewards[1], PenalizedLogPReward)

