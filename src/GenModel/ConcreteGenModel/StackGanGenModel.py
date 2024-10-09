import abc
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ImageData import ImageData
from GenerativeModelsData.StackGan2.StackGanUtils.model import G_NET
from GenerativeModelsData.StackGan2.StackGanUtils.config import cfg, cfg_from_file
from src.Logger.AbsLogger import AbsLogger
from torch.distributions.normal import Normal

import torch
import warnings


class StackGanGenModel(object, metaclass=abc.ABCMeta):
    """
        Base class incupsulating the required logic for StackGanv2 model
    """

    def __init__(self, config_file, checkpoint_file, gen_level, ngpu = 1):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.gen_level = gen_level
        self.device = 'cpu'

        cfg_from_file(self.config_file)
        self.model = G_NET()

        self.model = torch.nn.DataParallel(self.model, list(range(ngpu)))
        self.model.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device, weights_only=True))
        self.model.eval()
    
    def SetDevice(self, device) -> None:
        """
            Empty function.

            TODO:
                support parralel
        """
        
        warnings.warn("No other devices are supported for StackGan model", RuntimeWarning)
    
    def Generate(self, data: ActionData) -> ImageData:
        """
            Generates values based on provided action data
        """
        return ImageData(images = self.model(data.actions)[0][self.gen_level])


    def GetNoise(self, N:int = 1) -> ActionData:
        """
            Returns N samples of noise used for generation

            Returns:
                [N, D] tensor, N - number of noise vectors, D - dim of noise vectors
        """

        if N < 1:
            raise Exception(f'Generate noise number cannot be lower 1. Provided: {N}')
        
        # cfg_from_file(self.config_file)

        # action_dim = cfg.GAN.Z_DIM

        # return ActionData(actions=torch.normal(
        #                             torch.zeros((N, action_dim)), 
        #                             torch.ones((N, action_dim))
        #                                       )
        #                  )

        dist = self.GetDistribution()
        actions = []
        for i in range(N):
            actions.append(dist.sample())

        return ActionData(actions=torch.stack(actions, dim=0))
    
    def GetDistribution(self) -> Normal:
        """
            Returns the torch distribution object, corresponding to the 
            gen model
        """
        
        cfg_from_file(self.config_file)

        action_dim = cfg.GAN.Z_DIM

        dist = Normal(torch.tensor([0.0] * action_dim), torch.tensor([1.0] * action_dim), validate_args=None)
        
        return dist
         
    def GetMediaLogger(self) -> AbsLogger:
        """
            Returns a logger capable of logging the generated objects
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'StackGan model. \nConfig: {self.config_file}\nCheckpoint: {self.checkpoint_file}'