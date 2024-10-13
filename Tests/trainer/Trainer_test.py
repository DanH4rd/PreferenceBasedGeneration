import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

###################################################
###################################################
###################################################
###################################################

from torch.utils.tensorboard import SummaryWriter

from src.FeedbackSource.ConcreteFeedbackSource.RandomFeedbackSource import (
    RandomFeedbackSource,
)
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.Loss.ConcreteLoss.ActionRewardLoss import ActionRewardLoss
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.MetricsLogger.ConcreteMetricsLogger.TensorboardScalarLogger import (
    TensorboardScalarLogger,
)
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ConcreteTrainer.ptLightningTrainer import (
    ptLightningLatentWrapper,
    ptLightningModelWrapper,
    ptLightningTrainer,
)


class TestTrainer:

    gen_model = StackGanGenModel(
        config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
        checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
        gen_level=2,
    )

    def test_ptl_trainer_for_reward_model(self):

        nz = self.gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300

        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)

        prefLoss = PreferenceLoss(rewardModel=reward_model, decimals=None)

        writer = SummaryWriter(log_dir="Tests/trainer/runs_model")
        logger = TensorboardScalarLogger(name="pref_loss", writer=writer)
        prefLoss = LogLossDecorator(lossObject=prefLoss, logger=logger)

        reward_model = ptLightningModelWrapper(
            model=reward_model, loss_func_obj=prefLoss
        )

        trainer = ptLightningTrainer(model=reward_model, batch_size=2)

        feedbackSource = RandomFeedbackSource()
        dataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedbackSource)

        action_data, pref_data = dataGenerator.generate_preference_data(
            data=self.gen_model.sample_random_actions(N=5), limit=20
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

        assert len(logger.history["base"]) == (10 / 2) * 11
        assert len(logger.history["_epoch"]) == 11

    ############
    def test_ptl_trainer_for_latent(self):
        nz = self.gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300

        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)
        prefLoss = PreferenceLoss(rewardModel=reward_model, decimals=None)
        reward_model = ptLightningModelWrapper(
            model=reward_model, loss_func_obj=prefLoss
        )

        # reward_model_params_control = map(lambda x: x.data.clone(), reward_model.parameters())

        action = self.gen_model.sample_random_actions(N=1)

        action_clone = action.actions.clone()

        control_actions = self.gen_model.sample_random_actions(5)

        rewardLoss = ActionRewardLoss(rewardModel=reward_model)

        control_rewards = rewardLoss.calculate_loss(control_actions).detach()

        writer = SummaryWriter(log_dir="Tests/trainer/runs_latent")
        logger = TensorboardScalarLogger(name="action_reward_loss", writer=writer)
        rewardLoss = LogLossDecorator(lossObject=rewardLoss, logger=logger)

        model = ptLightningLatentWrapper(
            reward_model=reward_model, action=action, loss_func_obj=rewardLoss
        )

        trainer = ptLightningTrainer(model=model, batch_size=2)

        feedbackSource = RandomFeedbackSource()
        dataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedbackSource)

        action_data, pref_data = dataGenerator.generate_preference_data(
            data=self.gen_model.sample_random_actions(5), limit=20
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

        assert len(logger.history["base"]) == (10 / 2) * 11
        assert len(logger.history["_epoch"]) == 11

        # actions in ActionData correspond to optimised actions
        assert ~((action_clone - model.action) < 1e-10).all()
        assert ~((action_clone - action.actions) < 1e-10).all()

        post_rewards = rewardLoss.calculate_loss(control_actions).detach()

        # reward model did not change (reward values before and after latent opt are the same)
        assert ((control_rewards - post_rewards) < 1e-10).all()

        # reward vals checks fully replace reward model parametres check
        # post_reward_model_params= map(lambda x: x.data.clone(), reward_model.parameters())

        # # reward model params stayed unaffected
        # assert all( map(lambda control_and_post: ((control_and_post[0] - control_and_post[1]) < 1e-10).all(), zip(reward_model_params_control, post_reward_model_params)))
