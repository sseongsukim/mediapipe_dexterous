from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import numpy as np
from pytransform3d.rotations import quaternion_from_matrix, matrix_from_quaternion
from env.utils.filters import LPFilter, LPRotationFilter
from env.control.motion_control import PinocchioMotionControl
import env


def wrist_filters(
    wrist_mat: np.ndarray, pos_filter: LPFilter, rot_filter: LPRotationFilter
) -> np.ndarray:
    filtered_mat = wrist_mat.copy()
    filtered_mat[:3, 3] = pos_filter.next(wrist_mat[:3, 3])
    filtered_mat[:3, :3] = matrix_from_quaternion(
        rot_filter.next(quaternion_from_matrix(wrist_mat[:3, :3]))
    )
    return filtered_mat


class Controller(object):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.default_urdf_dir = Path(env.__path__[0]) / "assets"

        self.hand_dir = (
            Path(env.__path__[0]).parent
            / "dex_retargeting"
            / "assets"
            / "robots"
            / "hands"
        )

        self.dof = cfg.num_dof
        self._qpos = np.zeros(self.dof)

        self.ee_type = cfg.ee_type
        urdf_file = self.default_urdf_dir / Path(cfg.urdf_files)

        self.left_ee_controller = PinocchioMotionControl(
            urdf_file,
            cfg.left_wrist_name,
            np.array(cfg.left_arm_init),
            cfg.arm,
            arm_indices=self.cfg.left_arm_indices,
        )
        self.right_ee_controller = PinocchioMotionControl(
            urdf_file,
            cfg.right_wrist_name,
            np.array(cfg.right_arm_init),
            cfg.arm,
            arm_indices=self.cfg.right_arm_indices,
        )
        alpha = self.cfg.wrist_low_alpha
        self.left_wrist_pos_filter = LPFilter(alpha)
        self.left_wrist_rot_filter = LPRotationFilter(alpha)
        self.right_wrist_pos_filter = LPFilter(alpha)
        self.right_wrist_rot_filter = LPRotationFilter(alpha)

    def left_arm_pos(self, left_wrist):
        left_arm_pos = None
        if left_wrist is not None:
            left_wrist = wrist_filters(
                left_wrist,
                self.left_wrist_pos_filter,
                self.left_wrist_rot_filter,
            )
            left_arm_pos = self.left_ee_controller.control(
                left_wrist[:3, 3],
                left_wrist[:3, :3],
            )
        return left_arm_pos

    def right_arm_pos(self, right_wrist):
        right_arm_pos = None

        if right_wrist is not None:
            right_wrist = wrist_filters(
                right_wrist,
                self.right_wrist_pos_filter,
                self.right_wrist_rot_filter,
            )
            right_arm_pos = self.right_ee_controller.control(
                right_wrist[:3, 3],
                right_wrist[:3, :3],
            )
        return right_arm_pos
