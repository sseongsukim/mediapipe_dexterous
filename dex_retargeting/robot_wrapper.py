from typing import List

import numpy as np
import numpy.typing as npt
import pinocchio as pin

###### Model names ########
# inspire_hand : ['universe', 'index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint']
# ability_hand : ['universe', 'index_q1', 'index_q2', 'middle_q1', 'middle_q2', 'pinky_q1', 'pinky_q2', 'ring_q1', 'ring_q2', 'thumb_q1', 'thumb_q2']
#
###########################
MODEL_NAMES = {
    "inspire_hand": ['universe', 'index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint'],
    "ability_hand": ['universe', 'index_q1', 'index_q2', 'middle_q1', 'middle_q2', 'pinky_q1', 'pinky_q2', 'ring_q1', 'ring_q2', 'thumb_q1', 'thumb_q2'],
}


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()
        
        if "inspire" in urdf_path:
            self.model_names = MODEL_NAMES["inspire_hand"]
        elif "ability" in urdf_path:
            self.model_names = MODEL_NAMES["ability_hand"]
        else:
            raise ValueError
        

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError(f"Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return self.model_names
        # return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model_names) if nqs[i] > 0]
        # return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(f"{name} is not a link name. Valid link names: \n{self.link_names}")
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J