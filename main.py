"""Wires all components and runs the PbRL pipeline."""

from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from src.ActionDistribution.GreedyNormalActionDistribution import (
    GreedyNormalActionDistribution,
)
from src.DiscModel.StackGanDiscModel import StackGanDiscModel
from src.FeedbackSource.CosDistFeedback import CosDistFeedback
from src.FeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.FeedbackSource.HumanFeedback import HumanFeedback
from src.Filter.ScoreActionFilter import ScoreActionFilter
from src.GenModel.StackGanGenModel import StackGanGenModel
from src.Loss.ActionRewardLoss import ActionRewardLoss
from src.Loss.LogLossDecorator import LogLossDecorator
from src.Loss.PreferenceLoss import PreferenceLoss
from src.Memory.RoundsMemory import RoundsMemory
from src.MetricsLogger.TensorboardGridImageLogger import TensorboardGridImageLogger
from src.MetricsLogger.TensorboardImageLogger import TensorboardImageLogger
from src.MetricsLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.Pipeline.PbRLPipeline import PbRLPipeline
from src.PreferenceDataGenerator.BestActionTracker import BestActionTracker
from src.PreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)
from src.PreferenceDataGenerator.GraphPreferenceDataGeneration import (
    GraphPreferenceDataGeneration,
)
from src.RewardModel.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ptLightningTrainer import ptLightningTrainer
from src.Trainer.ptLightningWrappers import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
)

if __name__ == "__main__":
    # infrastructure
    tensorboard_writer = SummaryWriter(
        log_dir=f"logs\\{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
    )

    # metrics loggers
    pref_loss_logger = TensorboardScalarLogger(
        name="Loss/Preference Loss", writer=tensorboard_writer
    )
    action_loss_logger = TensorboardScalarLogger(
        name="Loss/Action Reward Loss", writer=tensorboard_writer
    )
    destination_handle_image_logger = TensorboardImageLogger(
        name="Image/Destination Image", writer=tensorboard_writer
    )
    control_max_logger = TensorboardGridImageLogger(
        name="Image/Control Max", writer=tensorboard_writer, nrow=3
    )
    control_min_logger = TensorboardGridImageLogger(
        name="Image/Control Min", writer=tensorboard_writer, nrow=3
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

    # set up feedback and preference generators
    # feedback_source = CosDistFeedback(
    #     target_image=Image.open(
    #         "GenerativeModelsData\\StackGan2\\target_images\\000387.jpg"
    #     ),
    #     th_min=0.01,
    #     th_max=0.75,
    #     device="cuda",
    #     gen_model=gen_model,
    # )

    feedback_source = HumanFeedback(
        window_name="Provide your preferences",
        gen_model=gen_model,
    )

    if isinstance(feedback_source, CosDistFeedback):
        tensorboard_writer.add_image(
            "Image/Target", pil_to_tensor(feedback_source.target_image), 0
        )

    preference_generator = GraphPreferenceDataGeneration(feedbackSource=feedback_source)
    preference_generator = BestActionTracker(prefDataGen=preference_generator)

    dummy_preference_generator = RandomPreferenceDataGenerator(
        feedbackSource=RandomFeedbackSource()
    )

    # set up losses
    preference_loss = LogLossDecorator(
        logger=pref_loss_logger,
        lossObject=PreferenceLoss(rewardModel=reward_model, decimals=None),
    )
    action_reward_loss = LogLossDecorator(
        logger=action_loss_logger, lossObject=ActionRewardLoss(rewardModel=reward_model)
    )

    # destination_action is shared mutable state: the same ActionData instance must
    # be passed to GreedyNormalActionDistribution, ptLightningLatentWrapper, and
    # PbRLPipeline. The latent trainer writes optimized tensors back to
    # destination_action.actions each epoch; the distribution reads
    # destination_action.actions[0] on update().
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

    # run pipeline
    pipeline = PbRLPipeline(
        config=PbRLPipeline.Configuration(
            rounds=15,
            samples_per_round=100,
            preference_limit_per_round=15,
            training_epochs_reward=10,
            training_epochs_latent=10,
            dummy_sample_size=10,
            preference_dummy_limit=100,
            control_sample_size=1000,
        ),
        gen_model=gen_model,
        action_dist=action_dist,
        destination_action=destination_action,
        preference_generator=preference_generator,
        dummy_preference_generator=dummy_preference_generator,
        memory=memory,
        model_trainer=model_trainer,
        latent_trainer=latent_trainer,
        sampling_filter=max_action_filter,
        control_max_filter=max_action_filter,
        control_min_filter=min_action_filter,
        round_image_logger=destination_handle_image_logger,
        control_max_logger=control_max_logger,
        control_min_logger=control_min_logger,
    )
    pipeline.run()
