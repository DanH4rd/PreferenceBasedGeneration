"""Wires all components and runs the PbRL pipeline."""

from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter

from src.ActionDistribution import (
    GreedyNormalActionDistribution,
)
from src.DataStructures import TrainableActionData
from src.DiscModel import StackGanDiscModel
from src.FeedbackSource import CosDistFeedback
from src.Filter import CompositeActionFilter, ScoreActionFilter
from src.GenModel import StackGanGenModel
from src.Loss import ActionRewardLoss, PreferenceLoss
from src.Memory import RoundsMemory
from src.Pipeline.PbRLPipeline import PbRLPipeline
from src.PreferenceDataGenerator import (
    BestActionTracker,
    GraphPreferenceDataGeneration,
)
from src.RewardModel import mlpRewardNetwork
from src.Trainer import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
    ptLightningTrainer,
)
from PIL import Image

if __name__ == "__main__":
    # infrastructure
    tensorboard_writer = SummaryWriter(
        log_dir=f"logs\\{datetime.now(tz=datetime.now().astimezone().tzinfo).strftime('%Y-%m-%d %H-%M-%S')}"
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

    # destination_action_trainable is shared mutable state: the same
    # TrainableActionData instance must be passed to GreedyNormalActionDistribution,
    # ptLightningLatentWrapper, HumanFeedback, and PbRLPipeline. It registers its
    # actions as an nn.Parameter, so the latent trainer optimises it in place
    # directly; the distribution reads its detached_actions[0] on update().
    destination_action_trainable = TrainableActionData.from_action_data(
        gen_model.sample_random_actions(N=1)
    )

    # feedback_source = HumanFeedback(
    #     window_name="Provide your preferences",
    #     gen_model=gen_model,
    #     destination_action_trainable=destination_action_trainable,
    # )

    feedback_source = CosDistFeedback(
        target_image=Image.open(
            "GenerativeModelsData\\StackGan2\\target_images\\000387.jpg"
        ),
        th_min=0.01,
        th_max=0.75,
        device="cuda",
        gen_model=gen_model,
    )

    preference_generator = BestActionTracker(
        prefDataGen=GraphPreferenceDataGeneration(feedbackSource=feedback_source)
    )

    # define action distribution

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

    # define trainers
    model_trainer = ptLightningTrainer(
        model=ptLightningModelWrapper(
            model=reward_model,
            loss_func_obj=PreferenceLoss(rewardModel=reward_model, decimals=None),
            loss_log_tensorboard_writer=tensorboard_writer,
            loss_log_name="Loss/Preference Loss",
        ),
        batch_size=20,
        # reward_model has no output activation/weight decay of its own; cap gradient
        # norm so a bad batch can't blow its weights up (was the actual source of the
        # NaN reward that later corrupted destination_action_trainable)
        gradient_clip_val=1.0,
    )

    latent_trainer = ptLightningTrainer(
        model=ptLightningLatentWrapper(
            trainable_action=destination_action_trainable,
            reward_model=reward_model,
            loss_func_obj=ActionRewardLoss(rewardModel=reward_model),
            loss_log_tensorboard_writer=tensorboard_writer,
            loss_log_name="Loss/Action Reward Loss",
        ),
        batch_size=20,
        # weight_decay pulls the action back toward the generator's N(0,1) prior each
        # step; gradient_clip_val caps per-step blowup. Both guard against unconstrained
        # reward-ascent walking the action out of distribution (degenerate/black images)
        # and eventually to NaN.
        optimizer_kwargs={"lr": 0.001, "weight_decay": 1e-2},
        gradient_clip_val=1.0,
    )

    # define memory
    memory = RoundsMemory(limit=10, discount_factor=0.99)

    # define sampling filter
    sampling_filter = CompositeActionFilter()
    sampling_filter.add_filter(
        ScoreActionFilter(
            mode="max", key=lambda x: reward_model.get_stable_rewards(x), limit=10
        )
    )
    # run pipeline
    pipeline = PbRLPipeline(
        config=PbRLPipeline.Configuration(
            rounds=15,
            samples_per_round=100,
            preference_limit_per_round=15,
            training_epochs_reward=50,
            training_epochs_latent=100,
            control_sample_size=1000,
        ),
        gen_model=gen_model,
        action_dist=action_dist,
        destination_action_trainable=destination_action_trainable,
        preference_generator=preference_generator,
        memory=memory,
        model_trainer=model_trainer,
        latent_trainer=latent_trainer,
        sampling_filter=sampling_filter,
        reward_model=reward_model,
        tensorboard_writer=tensorboard_writer,
    )
    pipeline.run()
