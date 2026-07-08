from torch.utils.tensorboard.writer import SummaryWriter

from src.FeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.Loss.ActionRewardLoss import ActionRewardLoss
from src.Loss.LogLossDecorator import LogLossDecorator
from src.Loss.PreferenceLoss import PreferenceLoss
from src.MetricsLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.PreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)
from src.RewardModel.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ptLightningTrainer import ptLightningTrainer
from src.Trainer.ptLightningWrappers import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
)


class TestTrainer:
    def test_ptl_trainer_for_reward_model(self, facade_gen_model, tmp_path):
        gen_model = facade_gen_model

        nz = gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300

        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)

        prefLoss = PreferenceLoss(rewardModel=reward_model, decimals=None)

        writer = SummaryWriter(log_dir=str(tmp_path / "runs_model"))
        logger = TensorboardScalarLogger(name="pref_loss", writer=writer)
        prefLoss = LogLossDecorator(lossObject=prefLoss, logger=logger)

        control_actions = gen_model.sample_random_actions(5)
        control_rewards = reward_model.get_stable_rewards(control_actions).detach()

        ptlreward_model = ptLightningModelWrapper(
            model=reward_model, loss_func_obj=prefLoss
        )

        trainer = ptLightningTrainer(model=ptlreward_model, batch_size=2)

        feedbackSource = RandomFeedbackSource()
        dataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedbackSource)

        action_data, pref_data = dataGenerator.generate_preference_data(
            data=gen_model.sample_random_actions(N=5), limit=20
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=1
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=5
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=5
        )

        writer.close()

        assert len(logger.history["base"]) == (10 / 2) * 11
        assert len(logger.history["_epoch"]) == 11

        post_rewards = reward_model.get_stable_rewards(control_actions).detach()
        # check if the model did change
        assert (abs(control_rewards - post_rewards) > 1e-10).all()

    ############
    def test_ptl_trainer_for_latent(self, facade_gen_model, tmp_path):
        gen_model = facade_gen_model

        nz = gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300

        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)
        prefLoss = PreferenceLoss(rewardModel=reward_model, decimals=None)
        reward_model = ptLightningModelWrapper(
            model=reward_model, loss_func_obj=prefLoss
        )

        # reward_model_params_control = map(lambda x: x.data.clone(), reward_model.parameters())

        action = gen_model.sample_random_actions(N=1)

        action_clone = action.actions.clone()

        control_actions = gen_model.sample_random_actions(5)

        rewardLoss = ActionRewardLoss(rewardModel=reward_model.model)

        control_rewards = rewardLoss.calculate_loss(control_actions).detach()

        writer = SummaryWriter(log_dir=str(tmp_path / "runs_latent"))
        logger = TensorboardScalarLogger(name="action_reward_loss", writer=writer)
        rewardLoss = LogLossDecorator(lossObject=rewardLoss, logger=logger)

        model = ptLightningLatentWrapper(
            reward_model=reward_model.model, action=action, loss_func_obj=rewardLoss
        )

        trainer = ptLightningTrainer(model=model, batch_size=2)

        feedbackSource = RandomFeedbackSource()
        dataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedbackSource)

        action_data, pref_data = dataGenerator.generate_preference_data(
            data=gen_model.sample_random_actions(5), limit=20
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=1
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=5
        )

        trainer.run_training(
            action_data=action_data, preference_data=pref_data, epochs=5
        )

        writer.close()

        assert len(logger.history["base"]) == (10 / 2) * 11
        assert len(logger.history["_epoch"]) == 11

        # actions in ActionData correspond to optimised actions
        assert ~((action_clone - model.action) < 1e-10).all()
        assert ~((action_clone - action.actions) < 1e-10).all()

        post_rewards = rewardLoss.calculate_loss(control_actions).detach()

        # reward model did not change (reward values before and after latent opt are the same)
        assert (abs(control_rewards - post_rewards) < 1e-10).all()
