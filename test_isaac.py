from omegaconf import OmegaConf, DictConfig
import hydra

from pathlib import Path

from envs.base import BaseEnv
from envs.teleoperation import MediapipeController
from dex_retargeting.mp_realsense import MediapipeRealsense
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


@hydra.main(config_path="configs", config_name="xarm7_ability")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    camera = MediapipeRealsense(**cfg.camera)
    depth_scale = camera.depth_scale
    env = BaseEnv(cfg)
    controller = MediapipeController(cfg, depth_scale)
    env.reset()
    k = 0

    while True:
        image_info = camera.get_frame()
        actions = controller.step(
            rgb_image=image_info["rgb_image"],
            depth_image_flipped=image_info["depth_image_flipped"],
        )
        env.step(actions)

        name_of_window = "SN:"
        # Display images
        cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name_of_window, image_info["image"])
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:
            break


if __name__ == "__main__":
    main()
