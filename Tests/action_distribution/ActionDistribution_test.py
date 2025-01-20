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


from src.ActionDistribution.ConcreteActionDistribution.SimpleActionDistribution import (
    SimpleActionDistribution,
)
from src.ActionDistribution.ConcreteActionDistribution.GreedyNormalActionDistribution import (
    GreedyNormalActionDistribution,
)
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel


class TestActionDistribution:

    def test_basic(self):
        model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )

        dist = SimpleActionDistribution(dist=model.get_input_noise_distribution())

        assert isinstance(dist.sample(N=1), ActionData)
        assert dist.sample(N=1).actions.shape[0] == 1
        assert dist.sample(N=4).actions.shape[0] == 4

    def test_greedy(self):
        model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )

        dist = GreedyNormalActionDistribution(dist=model.get_input_noise_distribution(),
                                              destination_action=model.sample_random_actions(N=1),
                                              e=0.9,
                                              decay_factor=0.8,
                                              omega2=0.5
                                              )

        assert isinstance(dist.sample(N=1), ActionData)
        assert dist.sample(N=1).actions.shape[0] == 1
        assert dist.sample(N=4).actions.shape[0] == 4
