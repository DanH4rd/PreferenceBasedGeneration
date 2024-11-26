import torch
from PIL import Image

from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.FeedbackSource.ConcreteFeedbackSource.CosDistFeedback import CosDistFeedback
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel


class TestFilter:

    def test_transformer_cos_feedback(self):

        target_image = Image.open("Tests\\feedback\\images\\ArtNouveaufacade79.jpeg")

        gen_model = StackGanGenModel(
            config_file="GenerativeModelsData\\StackGan2\\config\\facade_3stages_color.yml",
            checkpoint_file="GenerativeModelsData\\StackGan2\\checkpoints\\Facade v1.0\\netG_56500.pth",
            scale_level=0,
        )

        feedback = CosDistFeedback(
            target_image=target_image,
            th_min=0.01,
            th_max=0.75,
            device="cuda",
            gen_model=gen_model,
        )

        action_pairs = ActionPairsData(
            action_pairs=torch.stack(
                [
                    gen_model.sample_random_actions(N=5).actions,
                    gen_model.sample_random_actions(N=5).actions,
                ],
                dim=1,
            )
        )

        preference_data = feedback.generate_feedback(action_pairs_data=action_pairs)

        assert (
            action_pairs.action_pairs.shape[0]
            == preference_data.preference_pairs.shape[0]
        )
        assert (preference_data.preference_pairs.sum(dim=1) == 1).all()
