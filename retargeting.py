from env.base import BaseEnv
from env.configs import CONFIG_DICT
from env.teleoperation import MediapipeController

import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pyrealsense2 as rs
import cv2


class Retargeting(object):

    def __init__(self, cfg, env_cfg):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.mediapipe_controller = MediapipeController(cfg)
        self.setup_realsense()
        self.env = BaseEnv(env_cfg)

    def step(self):
        while True:
            images, rgb_image, depth_image_flipped = self.get_images()

            qpos = self.mediapipe_controller.step(
                rgb_image,
                depth_image_flipped,
                self.depth_scale,
            )
            self.env.step(qpos)
            name_of_window = "SN: " + str(self.device)
            cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(name_of_window, images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                break

    def get_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image, 1)
        color_image = np.asanyarray(color_frame.get_data())

        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        background_removed = np.where(
            (depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0),
            self.cfg.background_removed_color,
            color_image,
        )
        images = cv2.flip(background_removed, 1)
        color_image = cv2.flip(color_image, 1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return (images, color_images_rgb, depth_image_flipped)

    def setup_realsense(self):
        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(
                rs.camera_info.serial_number
            )
            connected_devices.append(detected_camera)
        self.device = connected_devices[0]
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        stream_res_x, stream_res_y = self.cfg.image_size
        rs_config.enable_device(self.device)
        rs_config.enable_stream(
            rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, self.cfg.fps
        )

        rs_config.enable_stream(
            rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, self.cfg.fps
        )
        profile = self.pipeline.start(rs_config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = self.cfg.clipping_distance_in_meters / self.depth_scale


@hydra.main(config_path="configs", config_name="xarm7_ability")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    env_cfg = CONFIG_DICT[cfg.env_config]
    Retargeting(cfg=cfg, env_cfg=env_cfg).step()


if __name__ == "__main__":
    main()
