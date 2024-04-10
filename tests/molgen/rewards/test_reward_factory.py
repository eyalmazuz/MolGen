from molgen.rewards.reward_factory import get_rewards
from molgen.rewards.functions.rdkit_rewards import QEDReward, PenalizedLogPReward
from molgen.rewards.functions.multi_reward import MultiReward


def test_get_single_reward_empty_dict():
    rewards_dict = {
        "QED": {}
    }
    
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, QEDReward)

    qed = reward("CCC")
    assert reward.scale is None
    assert abs(round(qed, 4) - 0.3854) < 0.0001


def test_get_single_reward_scale_in_dict():
    rewards_dict = {
            "QED": {"scale": lambda qed: 10 * qed}
    }
    
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, QEDReward)
    assert reward.scale is not None

    qed = reward("CCC")
    assert abs(round(qed, 4) - 3.854) < 0.001


def test_get_multiple_rewards_empty_list():
    rewards_dict = {
            "QED": {},
            "PlogP": {}
    }
    
    reward = get_rewards(rewards_dict)
    assert isinstance(reward, MultiReward)

    assert len(reward.rewards) == 2
    assert isinstance(reward.rewards[0], QEDReward)
    assert isinstance(reward.rewards[1], PenalizedLogPReward)
