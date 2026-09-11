import tkinter as tk
from dataclasses import dataclass

import torch
from PIL import ImageOps, ImageTk

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.DataStructures import (
    ActionData,
    ActionPairsData,
    PreferencePairsData,
    TrainableActionData,
)
from src.GenModel import StackGanGenModel

IMAGE_DISPLAY_SIZE = (400, 400)


class HumanFeedback(AbsFeedbackSource):
    """Generates preferences basing on cosinus similarity to the
    reference image calculated on image representations from
    a visual transformer
    """

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        window_name: str
        gen_model: StackGanGenModel
        destination_action_trainable: TrainableActionData

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return HumanFeedback(
            window_name=conf.window_name,
            gen_model=conf.gen_model,
            destination_action_trainable=conf.destination_action_trainable,
        )

    def __init__(
        self,
        window_name: str,
        gen_model: StackGanGenModel,
        destination_action_trainable: TrainableActionData,
    ):
        self.window_name = window_name
        self.gen_model = gen_model
        self.destination_action_trainable = destination_action_trainable

        # feedback interface elements
        self.reset_interface_elements()

    def reset_interface_elements(self):
        """Resets the feedback interface elements to None"""
        self.root = None
        self.left_label = None
        self.right_label = None
        self.target_label = None
        self._last_target_actions = None
        self.btn_left = None
        self.btn_right = None
        self.btn_equal = None
        self.btn_skip = None

    def reset_running_comparison_fields(self):
        self.user_preferences = []
        self.image_pairs = []

    def generate_feedback(
        self, action_pairs_data: ActionPairsData
    ) -> PreferencePairsData:
        self.reset_running_comparison_fields()

        action_data_l, action_data_r = action_pairs_data.get_split_actions()
        image_pairs_l = self.gen_model.generate(action_data_l).get_as_pil_images()
        image_pairs_r = self.gen_model.generate(action_data_r).get_as_pil_images()
        self.image_pairs = list(zip(image_pairs_l, image_pairs_r))

        target_actions = self.destination_action_trainable.detached_actions
        target_changed = self._last_target_actions is None or not torch.equal(
            target_actions, self._last_target_actions
        )

        if self.root is None:
            self.build_feedback_interface()
        if target_changed:
            target_action = ActionData(actions=target_actions)
            target_image = self.gen_model.generate(target_action).get_as_pil_images()[0]
            self.display_target_image(target_image)
            self._last_target_actions = target_actions
        self.display_image_pair(self.image_pairs[0])
        self.root.mainloop()

        accumulated_preferences = torch.tensor(self.user_preferences).to(
            action_pairs_data.action_pairs.device
        )

        return PreferencePairsData(preference_pairs=accumulated_preferences)

    def display_image_pair(self, image_pair):
        """Displays a pair of images in the feedback interface

        Args:
            image_pair (tuple): A tuple containing two PIL images (left and right)
        """
        assert self.left_label is not None
        assert self.right_label is not None

        left_image, right_image = image_pair

        # Fit into the display size, keeping aspect ratio, and convert to ImageTk format
        left_image_tk = ImageTk.PhotoImage(
            ImageOps.contain(left_image, IMAGE_DISPLAY_SIZE)
        )
        right_image_tk = ImageTk.PhotoImage(
            ImageOps.contain(right_image, IMAGE_DISPLAY_SIZE)
        )

        # Keep references to avoid garbage collection
        self._current_image_refs = (left_image_tk, right_image_tk)

        # Update the labels with the new images
        self.left_label.config(image=left_image_tk)
        self.right_label.config(image=right_image_tk)

    def display_target_image(self, target_image):
        """Displays the image for the current target (destination) action

        Args:
            target_image (Image.Image): PIL image of the target action
        """
        assert self.target_label is not None

        target_image_tk = ImageTk.PhotoImage(
            ImageOps.contain(target_image, IMAGE_DISPLAY_SIZE)
        )
        self._current_target_image_ref = target_image_tk
        self.target_label.config(image=target_image_tk)

    def accept_user_preference_btn_callback(self, preference: list[float]):
        """Accepts user preference and stores it in the user_preferences list

        Args:
            preference (list[float]): User preference pair, e.g. [1, 0] for left,
                [0, 1] for right, [0.5, 0.5] for equal, [0, 0] for skip
        """

        def add_preference():
            self.user_preferences.append(preference)
            if len(self.user_preferences) >= len(
                self.image_pairs
            ):  # Quit after collecting preferences for all image pairs
                self.close_feedback_interface()
            else:
                self.display_image_pair(self.image_pairs[len(self.user_preferences)])

        return add_preference

    def close_feedback_interface(self):
        """Stops the current query's event loop, keeping the window open
        for the next query"""
        assert self.root is not None
        self.root.quit()

    def build_feedback_interface(self):
        """Builds the persistent window for displaying images and
        collecting feedback. Called once; reused across queries"""
        self.root = tk.Tk()
        self.root.title(self.window_name)

        # Create frame for images
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        # Display images side by side: left, right, then the target column
        self.left_label = tk.Label(image_frame)
        self.left_label.pack(side=tk.LEFT, padx=10)

        self.right_label = tk.Label(image_frame)
        self.right_label.pack(side=tk.LEFT, padx=10)

        target_frame = tk.Frame(image_frame)
        target_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(target_frame, text="Current Target").pack()
        self.target_label = tk.Label(target_frame)
        self.target_label.pack()

        # Create frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Create 4 buttons
        self.btn_left = tk.Button(
            button_frame,
            text="Prefer Left",
            command=self.accept_user_preference_btn_callback([1, 0]),
        )
        self.btn_left.pack(side=tk.LEFT, padx=5)

        self.btn_right = tk.Button(
            button_frame,
            text="Prefer Right",
            command=self.accept_user_preference_btn_callback([0, 1]),
        )
        self.btn_right.pack(side=tk.LEFT, padx=5)

        self.btn_equal = tk.Button(
            button_frame,
            text="Equal",
            command=self.accept_user_preference_btn_callback([0.5, 0.5]),
        )
        self.btn_equal.pack(side=tk.LEFT, padx=5)

        self.btn_skip = tk.Button(
            button_frame,
            text="Skip",
            command=self.accept_user_preference_btn_callback([0, 0]),
        )
        self.btn_skip.pack(side=tk.LEFT, padx=5)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """

        return "Human feedback"
