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

import pytest

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel

generative_models = []

generative_models.append(
    StackGanGenModel(
        config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
        checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
        gen_level=2,
    )
)


class TestGeneratimeModel:

    @pytest.mark.parametrize(
        "model,model_name", zip(generative_models, map(str, generative_models))
    )
    def test_base(self, model, model_name):

        assert isinstance(model.sample_random_actions(N=1), ActionData)
        assert model.sample_random_actions(N=1).actions.shape[0] == 1
        assert model.sample_random_actions(N=4).actions.shape[0] == 4

        assert model.generate(model.sample_random_actions(N=1)).images.shape[0] == 1
        assert model.generate(model.sample_random_actions(N=1)).images.shape[1] == 3
        assert model.generate(model.sample_random_actions(N=5)).images.shape[0] == 5
