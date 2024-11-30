from typing import List, Dict, Optional

import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve


class PinocchioMotionControl(object):

    def __init__(
        self,
        urdf_file: str,
        wrist_name: str,
        arm_init_qpos: np.ndarray,
        arm_config: dict,
        arm_indices: Optional[List[int]] = [],
    ):
        self.arm_indices = arm_indices

        self.alpha = float(arm_config.out_lp_alpha)
        self.damp = float(arm_config.damp)
        self.eps = float(arm_config.eps)
        self.dt = float(arm_config.dt)

        self.model = pin.buildModelFromUrdf(str(urdf_file))
        self.dof = self.model.nq

        if arm_indices:
            locked_joint_indices = list(set(range(self.dof)) - set(self.arm_indices))
            locked_joint_indices = [idx + 1 for idx in locked_joint_indices]
            self.model = pin.buildReducedModel(
                self.model,
                locked_joint_indices,
                np.zeros(self.dof),
            )
        self.arm_dof = self.model.nq
        self.lower_limit = self.model.lowerPositionLimit
        self.high_limit = self.model.upperPositionLimit

        self.data = self.model.createData()
        self.wrist_id = self.model.getFrameId(wrist_name)

        self.qpos = arm_init_qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.wrist_pose = pin.updateFramePlacement(self.model, self.data, self.wrist_id)

    def control(self, target_pos, target_rot):
        oMdes = pin.SE3(target_rot, target_pos)

        qpos = self.qpos.copy()
        ik_qpos = qpos.copy()

        ik_qpos = self.ik_clik(ik_qpos, oMdes, self.wrist_id)
        qpos = ik_qpos.copy()

        self.qpos = pin.interpolate(self.model, self.qpos, qpos, self.alpha)
        self.qpos = qpos.copy()
        return self.qpos.copy()

    def ik_clik(self, qpos, oMdes, wrist_id, iter=1000):
        for _ in range(iter):
            pin.forwardKinematics(self.model, self.data, qpos)
            wrist_pose = pin.updateFramePlacement(self.model, self.data, wrist_id)
            iMd = wrist_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if norm(err) < self.eps:
                break
            jacobian = pin.computeFrameJacobian(self.model, self.data, qpos, wrist_id)
            jacobian = -np.dot(pin.Jlog6(iMd.inverse()), jacobian)

            v = -jacobian.T.dot(
                solve(jacobian.dot(jacobian.T) + self.damp * np.eye(6), err)
            )
            qpos = pin.integrate(self.model, qpos, v * self.dt)
        qpos = np.clip(qpos, self.lower_limit, self.high_limit)
        return qpos
