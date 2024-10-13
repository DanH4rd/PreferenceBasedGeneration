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


from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel

model = StackGanGenModel(
    config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
    checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
    gen_level=2,
)

assert isinstance(model.GetNoise(), ActionData)
assert model.GetNoise().actions.shape[0] == 1
assert model.GetNoise(N=4).actions.shape[0] == 4

assert model.Generate(model.GetNoise()).images.shape[0] == 1
assert model.Generate(model.GetNoise()).images.shape[1] == 3
assert model.Generate(model.GetNoise(N=5)).images.shape[0] == 5
