import PIL
from transformers import ViTImageProcessor, ViTModel
import torch
from src.FeedbackSource.AbsFeedbackSource import AbsFeedbackSource
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import PreferencePairsData
from src.GenModel.AbsGenModel import AbsGenModel

class CosDistFeedback(AbsFeedbackSource):
    """Generates preferences basing on cosinus similarity to the
    reference image calculated on image representations from
    a visual transformer
    """    

    def __init__(self, target_image:PIL.Image, th_min:float, th_max:float, device:str|None,
                 gen_model:AbsGenModel):
        self.th_min = th_min
        self.th_max = th_max

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device).eval()
        self.device = device

        self.target_image = target_image
        
        inputs = self.processor(images=self.target_image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        outputs = self.model(**inputs)
        target_cembed = outputs.last_hidden_state.detach()[:, 0]

        self.target_image_embed = target_cembed

        self.gen_model = gen_model

    def generate_feedback(
        self, action_pairs_data: ActionPairsData
    ) -> PreferencePairsData:
        
        # remake a list of pairs into a list of actions,
        # generate the list of images, calculate their
        # embedding vectors, calculate their cos
        # distances to target image and then
        # group values back into original pairs

        action_pairs = action_pairs_data.action_pairs
        action_data = action_pairs.view(action_pairs.shape[0] * action_pairs.shape[1],  action_pairs.shape[2])
        action_data = ActionData(actions=action_data)

        image_data = self.gen_model.generate(data=action_data)
        image_data.images = image_data.images.detach()

        inputs = self.processor(images=image_data.get_as_pil_images(), return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        outputs = self.model(**inputs)
        cos_embeds = outputs.last_hidden_state.detach()[:, 0]

        cos_pair_distances = (1 - torch.nn.functional.cosine_similarity(self.target_image_embed
                                                                , cos_embeds, dim=1
                                                                , eps=1e-6)).detach()
        cos_pair_distances = cos_pair_distances.view(action_pairs.shape[0],action_pairs.shape[1])

        cos_pair_diff = cos_pair_distances[:,0] - cos_pair_distances[:,1]
        
        preferences = torch.zeros(cos_pair_distances.shape).to(cos_pair_diff.device)

        # convert distances to preferences
        preferences[cos_pair_diff < 0] = torch.tensor([1.,0.]).to(cos_pair_diff.device)
        preferences[cos_pair_diff > 0] = torch.tensor([0., 1.]).to(cos_pair_diff.device)
        preferences[torch.abs(cos_pair_diff) < self.th_min] = torch.tensor([.5, .5]).to(cos_pair_diff.device)
        preferences[(cos_pair_distances > self.th_max).all(dim=1)] = torch.tensor([0.,0.]).to(cos_pair_diff.device)

        return PreferencePairsData(preference_pairs=preferences)

        
    def __str__(self) -> str:
            """Returns string describing the object

            Returns:
                str
            """

            return 'Transformer cos distance feedback'

