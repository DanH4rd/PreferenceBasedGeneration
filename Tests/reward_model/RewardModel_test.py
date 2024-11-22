import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

###################################################
###################################################
###################################################
###################################################


from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork


class TestRewardModel:
    def test_mlp_model(self):
        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )

        nz = gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300
        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)

        assert list(
            reward_model.get_rewards(gen_model.sample_random_actions(N=1)).shape
        ) == [1, 1]
        assert list(
            reward_model.get_rewards(gen_model.sample_random_actions(N=5)).shape
        ) == [5, 1]
