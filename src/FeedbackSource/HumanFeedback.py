import PIL
import torch
from dataclasses import dataclass

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsGenModel import AbsGenModel
from src.DataStructures.ActionData import ActionData
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.PreferencePairsData import PreferencePairsData

import tkinter as tk
from PIL import ImageTk


class HumanFeedback(AbsFeedbackSource):
    """Generates preferences basing on cosinus similarity to the
    reference image calculated on image representations from
    a visual transformer
    """

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        window_name: str
        gen_model: AbsGenModel

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return HumanFeedback(window_name=conf.window_name, gen_model=conf.gen_model)

    def __init__(self, window_name: str, gen_model: AbsGenModel):
        self.window_name = window_name
        self.gen_model = gen_model

        # feedback interface elements
        self.reset_interface_elements()

    def reset_interface_elements(self):
        """Resets the feedback interface elements to None"""
        self.root = None
        self.left_label = None
        self.right_label = None
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

        self.create_feedback_interface()

        accumulated_preferences = torch.tensor(self.user_preferences).to(
            action_pairs_data.action_pairs.device
        )

        return PreferencePairsData(preference_pairs=accumulated_preferences)

    def display_image_pair(self, image_pair):
        """Displays a pair of images in the feedback interface

        Args:
            image_pair (tuple): A tuple containing two PIL images (left and right)
        """
        left_image, right_image = image_pair

        # Convert PIL images to ImageTk format
        left_image_tk = ImageTk.PhotoImage(left_image)
        right_image_tk = ImageTk.PhotoImage(right_image)

        # Update the labels with the new images
        self.left_label.config(image=left_image_tk, width=384)
        self.left_label.image = (
            left_image_tk  # Keep a reference to avoid garbage collection
        )

        self.right_label.config(image=right_image_tk, width=384)
        self.right_label.image = (
            right_image_tk  # Keep a reference to avoid garbage collection
        )

    def accept_user_preference_btn_callback(self, preference: int):
        """Accepts user preference and stores it in the user_preferences list

        Args:
            preference (int): User preference (0 for left, 1 for right, 2 for equal, 3 for skip)
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
        """Closes the feedback interface window"""
        if self.root is not None:
            self.root.destroy()
            self.reset_interface_elements()

    def create_feedback_interface(self):
        """Creates a window for displaying images and collecting feedback"""
        self.root = tk.Tk()
        self.root.title(self.window_name)

        # Create frame for images
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        # Display images side by side
        self.left_label = tk.Label(image_frame)
        self.left_label.pack(side=tk.LEFT, padx=10)

        self.right_label = tk.Label(image_frame)
        self.right_label.pack(side=tk.RIGHT, padx=10)

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
            command=self.accept_user_preference_btn_callback([1, 1]),
        )
        self.btn_equal.pack(side=tk.LEFT, padx=5)

        self.btn_skip = tk.Button(
            button_frame,
            text="Skip",
            command=self.accept_user_preference_btn_callback([0, 0]),
        )
        self.btn_skip.pack(side=tk.LEFT, padx=5)

        self.display_image_pair(self.image_pairs[len(self.user_preferences)])
        self.root.mainloop()

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """

        return "Human feedback"
