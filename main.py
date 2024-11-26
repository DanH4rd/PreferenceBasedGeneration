"""implements the base pipeline of the system
"""

from datetime import datetime

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DiscModel.ConcreteDiscModel.StackGanDiscModel import StackGanDiscModel
from src.FeedbackSource.ConcreteFeedbackSource.CosDistFeedback import CosDistFeedback
from src.Filter.ConcreteActionFilter.ScoreActionFilter import ScoreActionFilter
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.Loss.ConcreteLoss.ActionRewardLoss import ActionRewardLoss
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.Memory.ConcreteMemory.RoundsMemory import RoundsMemory
from src.MetricsLogger.ConcreteMetricsLogger.TensorboardImageLogger import (
    TensorboardImageLogger,
)
from src.MetricsLogger.ConcreteMetricsLogger.TensorboardScalarLogger import (
    TensorboardScalarLogger,
)
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.BestActionTracker import (
    BestActionTracker,
)
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.GraphPreferenceDataGeneration import (
    GraphPreferenceDataGeneration,
)
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ConcreteTrainer.ptLightningTrainer import (
    ptLightningModelWrapper,
    ptLightningTrainer,
)

if __name__ == "__main__":
    rounds_number = 15

    reward_model = mlpRewardNetwork(input_dim=100, hidden_dim=100)

    gen_model = StackGanGenModel(
        config_file="GenerativeModelsData\\StackGan2\\config\\facade_3stages_color.yml",
        checkpoint_file="GenerativeModelsData\\StackGan2\\checkpoints\\Celeba v1.0\\netG_26000.pth",
        scale_level=0,
    )

    disc_model = StackGanDiscModel(
        config_file="GenerativeModelsData\\StackGan2\\config\\facade_3stages_color.yml",
        checkpoint_file="GenerativeModelsData\\StackGan2\\checkpoints\\Celeba v1.0\\netD0.pth",
        scale_level=0,
    )

    # feedback_source = RandomFeedbackSource()
    feedback_source = CosDistFeedback(
        target_image=Image.open(
            "GenerativeModelsData\\StackGan2\\target_images\\000387.jpg"
        ),
        th_min=0.01,
        th_max=0.75,
        device="cuda",
        gen_model=gen_model,
    )

    preference_generator = GraphPreferenceDataGeneration(feedbackSource=feedback_source)
    preference_generator = BestActionTracker(prefDataGen=preference_generator)

    tensorboard_writer = SummaryWriter(
        log_dir=f"logs\\{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}"
    )
    pref_loss_logger = TensorboardScalarLogger(
        name="Loss/Preference ", writer=tensorboard_writer
    )
    action_loss_logger = TensorboardScalarLogger(
        name="Loss/Action Reward", writer=tensorboard_writer
    )
    destination_handle_image_logger = TensorboardImageLogger(
        name="Image/Destination Image", writer=tensorboard_writer
    )

    if feedback_source.target_image != None:
        tensorboard_writer.add_image(
            "Image/Target", pil_to_tensor(feedback_source.target_image), 0
        )

    preference_loss = PreferenceLoss(rewardModel=reward_model, decimals=None)
    preference_loss = LogLossDecorator(
        logger=pref_loss_logger, lossObject=preference_loss
    )

    action_reward_loss = ActionRewardLoss(rewardModel=reward_model)
    action_reward_loss = LogLossDecorator(
        logger=action_loss_logger, lossObject=action_reward_loss
    )

    memory = RoundsMemory(limit=10, discount_factor=0.99)

    model_trainer = ptLightningTrainer(
        model=ptLightningModelWrapper(
            model=reward_model, loss_func_obj=preference_loss
        ),
        batch_size=20,
    )

    max_action_filter = ScoreActionFilter(
        mode="max", key=lambda x: reward_model.get_stable_rewards(x), limit=10
    )
    min_action_filter = ScoreActionFilter(
        mode="min", key=lambda x: reward_model.get_stable_rewards(x), limit=10
    )

    for r in range(rounds_number):
        sampled_actions = gen_model.sample_random_actions(100)
        sampled_actions = max_action_filter.filter(action_data=sampled_actions)
        action_data, pref_data = preference_generator.generate_preference_data(
            data=sampled_actions, limit=15
        )

        memory.add_data(
            ActionPairsPrefPairsContainer(
                action_pairs_data=action_data, pref_pairs_data=pref_data
            )
        )
        train_data = memory.get_data_from_memory()

        model_trainer.run_training(
            action_data=train_data.action_pairs_data,
            preference_data=train_data.pref_pairs_data,
            epochs=10,
        )

    control_actions = max_action_filter.filter(gen_model.sample_random_actions(1000))
    control_images = gen_model.generate(control_actions)
    control_image_grid = make_grid(control_images.images, nrow=3)
    tensorboard_writer.add_image("Image/Control Max", control_image_grid, 0)
    print("Distances and rewards for best images")
    print(feedback_source.get_cos_distances(actions=control_actions))
    print(reward_model.get_stable_rewards(data=control_actions)[:, 0])
    print()

    control_actions = min_action_filter.filter(gen_model.sample_random_actions(1000))
    control_images = gen_model.generate(control_actions)
    control_image_grid = make_grid(control_images.images, nrow=3)
    tensorboard_writer.add_image("Image/Control Min", control_image_grid, 0)
    print("Distances and rewards for worst images")
    print(feedback_source.get_cos_distances(actions=control_actions))
    print(reward_model.get_stable_rewards(data=control_actions)[:, 0])
    print()
