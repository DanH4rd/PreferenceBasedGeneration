import torch
import PIL


from src.DataStructures.AbsData import AbsData


class ImageData(AbsData):
    """Class for storing image data in defined format"""

    def __init__(self, images: torch.tensor):
        """

        Args:
            images (torch.tensor): [N, C, H, W] tensor
                N - number of images
                C - number of channels
                H - height
                W - width

        Raises:
            Exception: if tensor dimention length isn't 4
            Exception: if number of channels isn't equal to 1 or 3
        """

        self.images = images

        img_tensor_shape = self.images.shape

        if len(img_tensor_shape) != 4:
            raise Exception(
                f"Incorrect image tensor shape ({img_tensor_shape}). Tensor format should be NCHW (N - number of images, C - number of channels, H - height, W - width)"
            )

        if img_tensor_shape[1] != 1 and img_tensor_shape[1] != 3:
            raise Exception(
                f"Image has incorrect number of channels({img_tensor_shape[1]}). Image should be either RBG or monochrome"
            )
        
    def get_as_pil_images(self):
        """Returns stored image data as a list of PIL.Image objects
        """        

        pil_images = []

        for image_array in (self.images.detach().permute(0,2,3,1).cpu().numpy() * 255).astype('uint8'):
            pil_images.append(PIL.Image.fromarray(image_array))
        
        
        return pil_images

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Image Data"
