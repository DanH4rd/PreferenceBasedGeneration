import torch
from PIL import Image

from src.DataStructures.ActionPairsData import ActionPairsData
from src.FeedbackSource.CosDistFeedback import CosDistFeedback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFeedback:
    def test_transformer_cos_feedback(
        self, facade_gen_model_scale0, feedback_target_image_path
    ):
        target_image = Image.open(feedback_target_image_path)

        feedback = CosDistFeedback(
            target_image=target_image,
            th_min=0.01,
            th_max=0.75,
            device=DEVICE,
            gen_model=facade_gen_model_scale0,
        )

        action_pairs = ActionPairsData(
            action_pairs=torch.stack(
                [
                    facade_gen_model_scale0.sample_random_actions(N=5).actions,
                    facade_gen_model_scale0.sample_random_actions(N=5).actions,
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
