"""implements the base pipeline of the system
"""

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork
from src.FeedbackSource.ConcreteFeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.BestActionTracker import BestActionTracker
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.GraphPreferenceDataGeneration import GraphPreferenceDataGeneration
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.DiscModel.ConcreteDiscModel.StackGanDiscModel import StackGanDiscModel
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.Loss.ConcreteLoss.ActionRewardLoss import ActionRewardLoss
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.MetricsLogger.ConcreteMetricsLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.Memory.ConcreteMemory.RoundsMemory import RoundsMemory
from src.Trainer.ConcreteTrainer.ptLightningTrainer import ptLightningTrainer, ptLightningModelWrapper
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer

if __name__ == '__main__':
    rounds_number = 15

    reward_model = mlpRewardNetwork(input_dim = 100, hidden_dim=100)

    feedback_source = RandomFeedbackSource()
    preference_generator = GraphPreferenceDataGeneration(feedbackSource=feedback_source)
    preference_generator = BestActionTracker(prefDataGen=preference_generator)

    gen_model = StackGanGenModel(config_file='GenerativeModelsData\\StackGan2\\config\\facade_3stages_color.yml',
                                 checkpoint_file='GenerativeModelsData\\StackGan2\\checkpoints\\Facade v1.0\\netG_56500.pth',
                                 scale_level=0)
    
    disc_model = StackGanDiscModel(config_file='GenerativeModelsData\\StackGan2\\config\\facade_3stages_color.yml',
                                 checkpoint_file='GenerativeModelsData\\StackGan2\\checkpoints\\Facade v1.0\\netD0.pth',
                                 scale_level=0)
    

    tensorboard_writer =SummaryWriter(log_dir=f"logs\\{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}")
    pref_loss_logger = TensorboardScalarLogger(name='Preference Loss', writer=tensorboard_writer)
    action_loss_logger = TensorboardScalarLogger(name='action Reward Loss', writer=tensorboard_writer)


    preference_loss = PreferenceLoss(rewardModel=reward_model, decimals=None)
    preference_loss = LogLossDecorator(logger=pref_loss_logger, lossObject=preference_loss)

    action_reward_loss = ActionRewardLoss(rewardModel=reward_model)
    action_reward_loss = LogLossDecorator(logger=action_loss_logger, lossObject=action_reward_loss)

    memory = RoundsMemory(limit=10, discount_factor=0.99)

    model_trainer = ptLightningTrainer(model=ptLightningModelWrapper(model=reward_model, 
                                                                     loss_func_obj=preference_loss),
                                       batch_size=20)
    
    for r in range(rounds_number):
        sampled_actions = gen_model.sample_random_actions(10)
        action_data, pref_data = preference_generator.generate_preference_data(data=sampled_actions, limit=15)
        
        memory.add_data(ActionPairsPrefPairsContainer(action_pairs_data=action_data, pref_pairs_data=pref_data))
        train_data = memory.get_data_from_memory()

        model_trainer.run_training(action_data=train_data.action_pairs_data, 
                                   preference_data=train_data.pref_pairs_data,
                                   epochs=10)






