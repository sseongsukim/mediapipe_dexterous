import pyrealsense2 as rs
from typing import Optional

import numpy as np
from pathlib import Path
import os
import pickle
import cv2


class MediapipeRealsense(object):

    def __init__(
        self,
        resolution,
        fps,
        background_removed_color,
        clipping_distance_in_meters,
    ):
        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(
                rs.camera_info.serial_number
            )
            print(f"{detected_camera}")
            connected_devices.append(detected_camera)
        device = connected_devices[0]
        self.background_removed_color = background_removed_color
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        stream_res_x, stream_res_y = resolution
        stream_fps = fps
        rs_config.enable_stream(
            rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps
        )
        rs_config.enable_stream(
            rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps
        )
        self.profile = self.pipeline.start(rs_config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

    def get_frame(self):
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
            self.background_removed_color,
            color_image,
        )

        images = cv2.flip(background_removed, 1)
        color_image = cv2.flip(color_image, 1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return {
            "image": images,
            "rgb_image": color_images_rgb,
            "depth_image_flipped": depth_image_flipped,
        }
