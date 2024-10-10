
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

###################################################
###################################################
###################################################
###################################################


from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.DiscModel.ConcreteDiscModel.StackGanDiscModel import StackGanDiscModel
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData

gen_model = StackGanGenModel(
    config_file = './GenerativeModelsData/StackGan2/config/facade_3stages_color.yml',
    checkpoint_file = './GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth',
    gen_level=2
)

disc_model = StackGanDiscModel(
    config_file = './GenerativeModelsData/StackGan2/config/facade_3stages_color.yml',
    checkpoint_file = './GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netD2.pth',
    gen_level=2
)

assert(len(disc_model.Discriminate(gen_model.Generate(gen_model.GetNoise())).shape) == 1)
assert(disc_model.Discriminate(gen_model.Generate(gen_model.GetNoise())).shape[0] == 1)
assert(disc_model.Discriminate(gen_model.Generate(gen_model.GetNoise(5))).shape[0] == 5)