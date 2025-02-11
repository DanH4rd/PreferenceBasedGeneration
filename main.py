"""implements the base pipeline of the system
"""

from datetime import datetime

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from src.ActionDistribution.SimpleActionDistribution import SimpleActionDistribution
from src.ActionDistribution.GreedyNormalActionDistribution import GreedyNormalActionDistribution
from src.DataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DiscModel.StackGanDiscModel import StackGanDiscModel
from src.FeedbackSource.CosDistFeedback import CosDistFeedback
from src.FeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.Filter.ScoreActionFilter import ScoreActionFilter
from src.GenModel.StackGanGenModel import StackGanGenModel
from src.Loss.ActionRewardLoss import ActionRewardLoss
from src.Loss.LogLossDecorator import LogLossDecorator
from src.Loss.PreferenceLoss import PreferenceLoss
from src.Memory.RoundsMemory import RoundsMemory
from src.MetricsLogger.TensorboardImageLogger import TensorboardImageLogger
from src.MetricsLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.PreferenceDataGenerator.BestActionTracker import BestActionTracker
from src.PreferenceDataGenerator.RandomPreferenceDataGenerator import RandomPreferenceDataGenerator
from src.PreferenceDataGenerator.GraphPreferenceDataGeneration import (
    GraphPreferenceDataGeneration,
)
from src.RewardModel.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ptLightningTrainer import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
    ptLightningTrainer,
)

if __name__ == "__main__":
    rounds_number = 15

    # metrics loggers
    tensorboard_writer = SummaryWriter(
        log_dir=f"logs\\{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}"
    )
    pref_loss_logger = TensorboardScalarLogger(
        name="Loss/Preference Loss", writer=tensorboard_writer
    )
    action_loss_logger = TensorboardScalarLogger(
        name="Loss/Action Reward Loss", writer=tensorboard_writer
    )
    destination_handle_image_logger = TensorboardImageLogger(
        name="Image/Destination Image", writer=tensorboard_writer
    )

    # define ML models
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
    
    # set up feedback and pairs constructor
    feedback_source = CosDistFeedback(
        target_image=Image.open(
            "GenerativeModelsData\\StackGan2\\target_images\\000387.jpg"
        ),
        th_min=0.01,
        th_max=0.75,
        device="cuda",
        gen_model=gen_model,
    )

    if feedback_source.target_image != None:
        tensorboard_writer.add_image(
            "Image/Target", pil_to_tensor(feedback_source.target_image), 0
        )

    preference_generator = GraphPreferenceDataGeneration(feedbackSource=feedback_source)
    preference_generator = BestActionTracker(prefDataGen=preference_generator)

    dummy_preference_generator = RandomPreferenceDataGenerator(feedbackSource=RandomFeedbackSource())

    # set up losses
    preference_loss = LogLossDecorator(
        logger=pref_loss_logger,
        lossObject=PreferenceLoss(rewardModel=reward_model, decimals=None),
    )
    action_reward_loss = LogLossDecorator(
        logger=action_loss_logger, lossObject=ActionRewardLoss(rewardModel=reward_model)
    )

    # define memory and action distribution
    destination_action = gen_model.sample_random_actions(N=1)

    memory = RoundsMemory(limit=10, discount_factor=0.99)

    # action_dist = SimpleActionDistribution(
    #     dist=gen_model.get_input_noise_distribution()
    # )   
    
    action_dist = GreedyNormalActionDistribution(
            dist=gen_model.get_input_noise_distribution(),
            destination_action=destination_action,
            e=0.9,
            decay_factor=0.8,
            omega2=0.5,
        )


    tensorboard_writer.add_image(
        "Image/Starting Desc action",
        make_grid(gen_model.generate(destination_action).images, nrow=1),
        0,
    )

    # define trainers
    model_trainer = ptLightningTrainer(
        model=ptLightningModelWrapper(
            model=reward_model, loss_func_obj=preference_loss
        ),
        batch_size=20,
    )

    latent_trainer = ptLightningTrainer(
        model=ptLightningLatentWrapper(
            action=destination_action,
            reward_model=reward_model,
            loss_func_obj=action_reward_loss,
        ),
        batch_size=20,
    )

    # define filters
    max_action_filter = ScoreActionFilter(
        mode="max", key=lambda x: reward_model.get_stable_rewards(x), limit=10
    )
    min_action_filter = ScoreActionFilter(
        mode="min", key=lambda x: reward_model.get_stable_rewards(x), limit=10
    )

    for r in range(rounds_number):
        sampled_actions = action_dist.sample(100)
        sampled_actions = max_action_filter.filter(action_data=sampled_actions)
        sampled_actions.append(destination_action.actions.detach()
        )
        
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

        dummy_action_data, dummy_pref_data = (
            dummy_preference_generator.generate_preference_data(
                data=gen_model.sample_random_actions(10), limit=100
            )
        )

        action_dist.update(None)

        latent_trainer.run_training(
            action_data=dummy_action_data, preference_data=dummy_pref_data, epochs=10
        )

        destination_handle_image_logger.log(gen_model.generate(destination_action))

    control_actions = max_action_filter.filter(gen_model.sample_random_actions(1000))
    control_images = gen_model.generate(control_actions)
    control_image_grid = make_grid(control_images.images, nrow=3)
    tensorboard_writer.add_image("Image/Control Max", control_image_grid, 0)
    # print("Distances and rewards for best images")
    # print(feedback_source.get_cos_distances(actions=control_actions))
    # print(reward_model.get_stable_rewards(data=control_actions)[:, 0])
    # print()

    control_actions = min_action_filter.filter(gen_model.sample_random_actions(1000))
    control_images = gen_model.generate(control_actions)
    control_image_grid = make_grid(control_images.images, nrow=3)
    tensorboard_writer.add_image("Image/Control Min", control_image_grid, 0)
    # print("Distances and rewards for worst images")
    # print(feedback_source.get_cos_distances(actions=control_actions))
    # print(reward_model.get_stable_rewards(data=control_actions)[:, 0])
    # print()
