# Preference Based Generation system

This project is a redesign of my master thesis, inspired primarily by "Preference-Based Image Generation" article. 

(Hadi Kazemi, Fariborz Taherkhani, and Nasser M. Nasrabadi. Preference-based
image generation. In 2020 IEEE Winter Conference on Applications of Computer
Vision (WACV), pages 3393–3402, 2020. Link: https://www.overleaf.com/project/654a7b69e4ff0a4650ea3272)

Currently it is work in progress and it has implementation of the majority of the required components. System design is focused to work with pairwise preferences, while prioritising the ability to work with different medias.

A more detailed description of the system and its components will be added later.

# Implemented on a system:

* Windows 10
* Python 3.12
* CUDA 10.8

# How to run

Currently the project is hard coded to work with StackGan-v2 in unconditioned settings (Original at https://github.com/hanzhanggit/StackGAN-v2/issues, version refactored for later python versions and used during development of this project at https://github.com/rightlit/StackGAN-v2-rev).

When trained a StackGanV2 change the checkpoints paths for models at main.py (`gen_model` and `disc_model` variables). 

Project does not support human feedback at current version and uses artificial one, that requires a target image to be set. For this choose an image you desire an artificial user to strive to and set its path at of `feedback_source` variably at `main.py`.

As a final step install dependencies (preferably using venv) and simply run `python main.py`. You may inspect metrics and generation results using tensorboard at `localhost:6006`.

# Working with other models/media

To work with other image generation models or with other media generators you must define your own generator and discriminator classes that return values expected by other components. Additionally, if you wish to log the generated media as well, you must define your own logger as well.

# How it works

## Basic explanation

What the system attempts to do is to look for the best fitting image in the generator's knowledge. It does so by navigating through the noise space, on which the generator operates. System learns to score the noise vectors based on human (or artificial) feedback and then uses this score to move towards the direction of most gain. The starting point is a random noise vector from generator's noise distribution. System operates until the desired image is reached or the maximum number of iterations is reached.

## Detailed explanation

At the beginning of operation we define the destination action - a noise vector which is the starting point of our search. We also set up our reward model for actions (noise vectors), generator model and discriminator model.

The work of the system is devided into rounds. During each round first we randomly sample a set number of actions, then we filter them out to leave only a subset of them (at the general implementation we choose actions with highest rewards from the reward model). Next we combine them into pairs and present to our human or artificial user to choose which of the two is better than the other (or if they both good or bad).

Next the gained feedback is added to the memory and then data from the said memory is used to optimise the reward model for a set amount of epochs. As a loss function a cross entropy loss is used.

With the updated reward model we optimise the destination action for the set amoun of epochs. We treat the noise vector as parameters and as a loss function we use a negative reward from the reward model, thus maximising the reward for the destination action.

After that the round ends and the next one starts.

Different strategies may be applied at different stages of a round, which would be described as the project evolves.
