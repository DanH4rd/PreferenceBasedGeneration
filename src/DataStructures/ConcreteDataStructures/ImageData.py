from src.DataStructures.AbsData import AbsData
import torch

class ImageData(AbsData):
    """
        Class for storing image data as tensor
    """

    def __init__(self, images:torch.tensor):
        """
            Parametres:

                images -  [N, C, H, W] tensor
                    N - number of images
                    C - number of channels
                    H - height
                    W - width
        """
        self.images = images

        img_tensor_shape = self.images.shape

        if len(img_tensor_shape) != 4:
            raise Exception(f'Incorrect image tensor shape ({img_tensor_shape}). Tensor format should be NCHW (N - number of images, C - number of channels, H - height, W - width)')

    
        if img_tensor_shape[1] != 1 and img_tensor_shape[1] != 3:
            raise Exception(f'Image has incorrect number of channels({img_tensor_shape[1]}). Image should be either RBG or monochrome')

    
    def __str__(self) -> str:
       return "Image Data"