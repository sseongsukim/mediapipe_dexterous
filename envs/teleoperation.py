import numpy as np
import pyrealsense2 as rs
import cv2
import mediapipe as mp

import dex_retargeting
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.dual_hand_detector import DualHandDector

import transforms3d

from pathlib import Path
import os

from pynput.keyboard import Key, Listener


class MediapipeController(object):

    def __init__(self, cfg, depth_scale):
        self.cfg = cfg
        self.depth_scale = depth_scale
        self.action_dim = cfg.action_dim

        self.left_joint_action_indices = cfg.left_joint_indices
        self.left_hand_action_indices = cfg.left_ee_indices
        self.right_joint_action_indices = cfg.right_joint_indices
        self.right_hand_action_indices = cfg.right_ee_indices

        robot_dir = Path(dex_retargeting.__path__[0]) / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        right_config_path = Path(dex_retargeting.__path__[0]) / cfg.right_config_path
        left_config_path = Path(dex_retargeting.__path__[0]) / cfg.left_config_path

        right_retargeting = RetargetingConfig.load_from_file(
            str(right_config_path)
        ).build()
        left_retargeting = RetargetingConfig.load_from_file(
            str(left_config_path)
        ).build()

        self.detector = DualHandDector(
            left_retargeting=left_retargeting,
            right_retargeting=right_retargeting,
            depth_scale=depth_scale,
            selfie=False,
        )
        self.reset_actions()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def reset_actions(self):
        self.actions = np.zeros(self.action_dim, dtype=np.float32)

    def step(self, rgb_image, depth_image_flipped):
        actions = np.zeros(self.action_dim, dtype=np.float32)
        detect_info = self.detector.detect(rgb_image, depth_image_flipped)

        if detect_info is None:
            print("detect info is None")
            return self.actions.copy()
        else:
            if "Left" in detect_info.keys():
                actions[self.left_hand_action_indices] = detect_info["Left"]["qpos"]
                left_quat = transforms3d.quaternions.mat2quat(
                    detect_info["Left"]["wrist_rot"]
                )
                left_quat[2] -= 0.47
                left_quat[1] -= 0.49
                left_quat[0] -= 0.53
                left_quat[3] -= 0.4
                left_pos = detect_info["Left"]["wrist_pos"]
                left_pos[0] += 0.5
                left_pos[1] += 0.5
                left_pos[2] -= 0.4
                actions[:3] = left_pos
                actions[3:7] = left_quat
                print(left_quat)

            if "Right" in detect_info.keys():
                actions[self.right_hand_action_indices] = detect_info["Right"]["qpos"]
                right_quat = np.array(
                    list(
                        transforms3d.quaternions.mat2quat(
                            detect_info["Right"]["wrist_rot"]
                        )
                    )
                )
                right_quat = right_quat[[2, 1, 0, 3]]
                right_quat[0] += 0.47
                right_quat[1] + 0.55
                right_quat[2] -= 0.46
                right_quat[3] -= 0.5
                print(right_quat)
                actions[20:24] = right_quat
                right_pos = detect_info["Right"]["wrist_pos"]
                right_pos[0] += 0.5
                right_pos[1] += 0.5
                right_pos[2] -= 0.4
                actions[17:20] = right_pos

        self.actions = actions.copy()
        return actions

    def on_press(self, key):
        try:
            k = key.char
            if k == "q":
                self.right_arm[0] += self.k
            elif k == "a":
                self.right_arm[0] -= self.k
            elif k == "w":
                self.right_arm[1] += self.k
            elif k == "s":
                self.right_arm[1] -= self.k
            elif k == "e":
                self.right_arm[2] += self.k
            elif k == "d":
                self.right_arm[2] -= self.k
            elif k == "r":
                self.right_arm[3] += self.k
            elif k == "f":
                self.right_arm[3] -= self.k
            elif k == "t":
                self.right_arm[4] += self.k
            elif k == "g":
                self.right_arm[4] -= self.k
            elif k == "y":
                self.right_arm[5] += self.k
            elif k == "h":
                self.right_arm[5] -= self.k
            elif k == "u":
                self.right_arm[6] += self.k
            elif k == "j":
                self.right_arm[6] -= self.k

        except AttributeError as e:
            print(f"Special key pressed: {key}")

    def on_release(self, key):
        if key == Key.esc:
            # Stop listener
            return False
