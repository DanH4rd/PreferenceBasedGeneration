"""Wires all components and runs the PbRL pipeline."""

from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from src.ActionDistribution import (
    GreedyNormalActionDistribution,
)
from src.DataStructures import ActionData, TrainableActionData
from src.DiscModel import StackGanDiscModel
from src.FeedbackSource import CosDistFeedback, HumanFeedback, RandomFeedbackSource
from src.Filter import ScoreActionFilter
from src.GenModel import StackGanGenModel
from src.Loss import ActionRewardLoss, LogLossDecorator, PreferenceLoss
from src.Memory import RoundsMemory
from src.MetricsLogger import (
    TensorboardGridImageLogger,
    TensorboardImageLogger,
    TensorboardScalarLogger,
)
from src.Pipeline.PbRLPipeline import PbRLPipeline
from src.PreferenceDataGenerator import (
    BestActionTracker,
    GraphPreferenceDataGeneration,
    RandomPreferenceDataGenerator,
)
from src.RewardModel import mlpRewardNetwork
from src.Trainer import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
    ptLightningTrainer,
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

    # destination_action_trainable is shared mutable state: the same
    # TrainableActionData instance must be passed to GreedyNormalActionDistribution,
    # ptLightningLatentWrapper, and PbRLPipeline. It registers its actions as an
    # nn.Parameter, so the latent trainer optimises it in place directly; the
    # distribution reads its detached_actions[0] on update().
    destination_action_trainable = TrainableActionData.from_action_data(
        gen_model.sample_random_actions(N=1)
    )

    memory = RoundsMemory(limit=10, discount_factor=0.99)

    # action_dist = SimpleActionDistribution(
    #     dist=gen_model.get_input_noise_distribution()
    # )

    action_dist = GreedyNormalActionDistribution(
        dist=gen_model.get_input_noise_distribution(),
        destination_action_trainable=destination_action_trainable,
        e=0.9,
        decay_factor=0.8,
        omega2=0.5,
    )

    tensorboard_writer.add_image(
        "Image/Starting Desc action",
        make_grid(
            gen_model.generate(
                ActionData(actions=destination_action_trainable.detached_actions)
            ).images,
            nrow=1,
        ),
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
            trainable_action=destination_action_trainable,
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
        destination_action_trainable=destination_action_trainable,
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
