from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import torch
from collections import defaultdict
import math
import os
from pathlib import Path
import envs
import numpy as np
import gymnasium as gym


from envs.utils.torch_utils import to_numpy, to_torch


@torch.jit.script
def ik_controller(jacobian, action):
    damping = 0.1
    jacobian_T = torch.transpose(jacobian, 1, 2)
    lmbda = torch.eye(6) * (damping**2)
    u = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lmbda) @ action
    return u


def get_dpose(init_hand_pos, init_hand_quat, arms, hand_pos, hand_quat):
    goal_pos = init_hand_pos + arms[:, :3]
    pos_error = goal_pos - hand_pos

    goal_quat = init_hand_quat + arms[:, 3:]
    quat_error = orientation_error(goal_quat, hand_quat)

    dpose = torch.cat([pos_error, quat_error], dim=-1).unsqueeze(-1)
    return dpose


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class BaseEnv(gym.Env):

    def __init__(self, cfg: dict):
        super(BaseEnv, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.headless = cfg.headless
        self.num_envs = cfg.num_envs
        self.action_repeat = cfg.action_repeat
        self.action_dim = cfg.action_dim
        self.gym = gymapi.acquire_gym()

        self.sim_params = self.default_sim_params()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        self.create_env()

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            focus_envs = self.env[0]
            self.gym.viewer_camera_look_at(self.viewer, focus_envs, cam_pos, cam_target)

        self.camera_setup()

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(
            self.num_envs, self.left_num_dofs + self.right_num_dofs
        )
        self.dof_vel = self.dof_states[:, 1].view(
            self.num_envs, self.left_num_dofs + self.right_num_dofs
        )

        _forces = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim))
        self.forces = _forces.view(
            self.num_envs, self.left_num_dofs + self.right_num_dofs
        )

        _left_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_robot")
        self.left_jacobian = gymtorch.wrap_tensor(_left_jacobian)
        self.left_base_jacobian = self.left_jacobian[:, self.left_hand_idx - 1, :, :7]

        _right_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "right_robot")
        self.right_jacobian = gymtorch.wrap_tensor(_right_jacobian)
        self.right_base_jacobian = self.right_jacobian[
            :, self.right_hand_idx - 1, :, :7
        ]

        self.gym.prepare_sim(self.sim)

        self.left_joint_indices = self.cfg.left_joint_indices
        self.left_ee_indices = self.cfg.left_ee_indices
        self.right_joint_indices = self.cfg.right_joint_indices
        self.right_ee_indices = self.cfg.right_ee_indices

        self.down_q = (
            torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])])
            .to(self.device)
            .view((self.num_envs, 4))
        )

        self.refresh()
        (
            self.init_left_hand_pos,
            self.init_left_hand_quat,
            self.init_right_hand_pos,
            self.init_right_hand_quat,
        ) = self.get_hand_poses()

    def get_hand_poses(self):
        left_hand_pos = self.rb_states[self.left_hand_indices, 0:3]
        left_hand_quat = self.rb_states[self.left_hand_indices, 3:7]

        left_base_pos = self.rb_states[self.left_base_indices, 0:3]

        right_hand_pos = self.rb_states[self.right_hand_indices, 0:3]
        right_hand_quat = self.rb_states[self.right_hand_indices, 3:7]

        right_base_pos = self.rb_states[self.right_base_indices, 0:3]

        return (
            left_hand_pos - left_base_pos,
            left_hand_quat,
            right_hand_pos - right_base_pos,
            right_hand_quat,
        )

    def ik_info(self):

        left_joint_pos = self.dof_pos[:, self.left_joint_indices].clone()
        left_jacobian = self.left_base_jacobian.clone()

        right_joint_pos = self.dof_pos[:, self.right_joint_indices].clone()
        right_jacobian = self.right_base_jacobian.clone()

        return left_joint_pos, left_jacobian, right_joint_pos, right_jacobian

    def step(self, actions: np.ndarray):
        actions = np.expand_dims(actions, 0)

        left_joint_actions = to_torch(
            actions[:, self.left_joint_indices], device=self.device
        )
        right_joint_actions = to_torch(
            actions[:, self.right_joint_indices], device=self.device
        )
        left_hand_actions = to_torch(
            actions[:, self.left_ee_indices], device=self.device
        )
        right_hand_actions = to_torch(
            actions[:, self.right_ee_indices], device=self.device
        )
        for _ in range(self.action_repeat):
            self.refresh()
            left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat = (
                self.get_hand_poses()
            )

            pos_actions = torch.zeros(
                size=(self.num_envs, self.left_num_dofs + self.right_num_dofs),
                dtype=torch.float32,
                device=self.device,
            )

            left_joint_pos, left_jacobian, right_joint_pos, right_jacobian = (
                self.ik_info()
            )

            left_dpose = get_dpose(
                self.init_left_hand_pos,
                self.init_left_hand_quat,
                left_joint_actions,
                left_hand_pos,
                left_hand_quat,
            )
            right_dpose = get_dpose(
                self.init_right_hand_pos,
                self.init_right_hand_quat,
                right_joint_actions,
                right_hand_pos,
                right_hand_quat,
            )
            # Arm
            pos_actions[:, self.left_joint_indices] = left_joint_pos + ik_controller(
                left_jacobian, left_dpose
            ).view(self.num_envs, 7)
            pos_actions[:, self.right_joint_indices] = right_joint_pos + ik_controller(
                right_jacobian, right_dpose
            ).view(self.num_envs, 7)
            # Hand
            pos_actions[:, self.left_ee_indices] = left_hand_actions
            pos_actions[:, self.right_ee_indices] = right_hand_actions

            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(pos_actions)
            )

            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

        self.gym.end_access_image_tensors(self.sim)
        self.env_steps += 1

    def reset(self):
        self.initialize_robots()
        self.env_steps = np.zeros(self.num_envs, dtype=np.uint32)
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torch.zeros_like(self.dof_pos))
        )
        self.refresh()

    def initialize_robots(self):
        self.dof_pos[:, : self.left_num_dofs + self.right_num_dofs] = torch.tensor(
            self.robot_default_dof_pos,
            device=self.device,
            dtype=torch.float32,
        )
        zero_vel = [0 for _ in range(len(self.robot_default_dof_pos))]
        self.dof_vel[:, : self.left_num_dofs + self.right_num_dofs] = torch.tensor(
            zero_vel, dtype=torch.float32, device=self.device
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.robot_all_indices_t),
            len(self.robot_all_indices_t),
        )

    def create_env(self):

        left_asset_options = self.default_asset_options()
        asset_roots = Path(envs.__path__[0]) / "assets"
        self.left_robot_asset = self.gym.load_asset(
            self.sim,
            str(asset_roots),
            self.cfg.left_urdf,
            left_asset_options,
        )

        right_asset_options = self.default_asset_options()
        self.right_robot_asset = self.gym.load_asset(
            self.sim,
            str(asset_roots),
            self.cfg.right_urdf,
            right_asset_options,
        )

        row_length = int(math.sqrt(self.num_envs))
        env_lower = gymapi.Vec3(-self.cfg.spacing, -self.cfg.spacing, 0.0)
        env_upper = gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing)

        self.left_robot_pose = gymapi.Transform()
        self.left_robot_pose.p = gymapi.Vec3(0.0, 0.35, 0.0)
        self.left_robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.right_robot_pose = gymapi.Transform()
        self.right_robot_pose.p = gymapi.Vec3(0.0, -0.35, 0.0)
        self.right_robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        left_robot_link_dict = self.gym.get_asset_rigid_body_dict(self.left_robot_asset)
        self.left_hand_idx = left_robot_link_dict[self.cfg.ee_name]
        self.left_base_idx = left_robot_link_dict[self.cfg.base]

        right_robot_link_dict = self.gym.get_asset_rigid_body_dict(
            self.right_robot_asset
        )
        self.right_hand_idx = right_robot_link_dict[self.cfg.ee_name]
        self.right_base_idx = right_robot_link_dict[self.cfg.base]

        left_dof_props = self.gym.get_asset_dof_properties(self.left_robot_asset)
        right_dof_props = self.gym.get_asset_dof_properties(self.right_robot_asset)

        self.left_dof = self.gym.get_asset_dof_count(self.left_robot_asset)
        self.right_dof = self.gym.get_asset_dof_count(self.right_robot_asset)

        for i in range(self.left_dof):
            left_dof_props["stiffness"][i] = 400.0
            left_dof_props["damping"][i] = 40.0

        for i in range(self.right_dof):
            right_dof_props["stiffness"][i] = 400.0
            right_dof_props["damping"][i] = 40.0

        self.left_default_dof_pos = np.array(self.cfg.left_arm_init)
        left_default_dof_pos = np.zeros(self.left_dof, dtype=gymapi.DofState.dtype)
        left_default_dof_pos["pos"] = self.left_default_dof_pos

        self.right_default_dof_pos = np.array(self.cfg.right_arm_init)
        right_default_dof_pos = np.zeros(self.right_dof, dtype=gymapi.DofState.dtype)
        right_default_dof_pos["pos"] = self.right_default_dof_pos

        self.robot_default_dof_pos = np.concatenate(
            [self.left_default_dof_pos, self.right_default_dof_pos]
        )

        self.env = []
        self.env_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.left_hand_indices = []
        self.left_base_indices = []

        self.right_hand_indices = []
        self.right_base_indices = []

        self.left_robot_handles = []
        self.right_robot_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, row_length)
            self.env.append(env)

            left_robot_handle = self.gym.create_actor(
                env, self.left_robot_asset, self.left_robot_pose, "left_robot", i, 0
            )
            self.left_num_dofs = self.gym.get_actor_dof_count(env, left_robot_handle)

            right_robot_handle = self.gym.create_actor(
                env, self.right_robot_asset, self.right_robot_pose, "right_robot", i, 0
            )
            self.right_num_dofs = self.gym.get_actor_dof_count(env, right_robot_handle)

            self.gym.enable_actor_dof_force_sensors(env, left_robot_handle)
            self.gym.enable_actor_dof_force_sensors(env, right_robot_handle)

            self.left_robot_handles.append(left_robot_handle)
            self.right_robot_handles.append(right_robot_handle)

            self.left_hand_indices.append(
                self.gym.get_actor_rigid_body_index(
                    env, left_robot_handle, self.left_hand_idx, gymapi.DOMAIN_SIM
                )
            )
            self.right_hand_indices.append(
                self.gym.get_actor_rigid_body_index(
                    env, right_robot_handle, self.right_hand_idx, gymapi.DOMAIN_SIM
                )
            )
            self.left_base_indices.append(
                self.gym.get_actor_rigid_body_index(
                    env, left_robot_handle, self.left_base_idx, gymapi.DOMAIN_SIM
                )
            )
            self.right_base_indices.append(
                self.gym.get_actor_rigid_body_index(
                    env, right_robot_handle, self.right_base_idx, gymapi.DOMAIN_SIM
                )
            )

            self.gym.set_actor_dof_properties(env, left_robot_handle, left_dof_props)
            self.gym.set_actor_dof_properties(env, right_robot_handle, right_dof_props)

            self.gym.set_actor_dof_states(
                env, left_robot_handle, left_default_dof_pos, gymapi.STATE_ALL
            )
            self.gym.set_actor_dof_states(
                env, right_robot_handle, right_default_dof_pos, gymapi.STATE_ALL
            )

        self.robot_all_indices = []

        for i in range(self.num_envs):
            self.robot_all_indices.append(
                self.gym.find_actor_index(self.env[i], "left_robot", gymapi.DOMAIN_SIM)
            )
            self.robot_all_indices.append(
                self.gym.find_actor_index(self.env[i], "right_robot", gymapi.DOMAIN_SIM)
            )

        self.robot_all_indices_t = torch.tensor(
            self.robot_all_indices, dtype=torch.int32, device=self.device
        )

    def render(self, env_idx=0):
        for k, v in self.cameras.items():
            frame = to_numpy(v[env_idx][:, :, :-1].clone())
        return frame

    def camera_setup(self):
        front_camera = gymapi.IMAGE_COLOR
        self.cameras = defaultdict(list)
        for i in range(self.num_envs):
            self.camera_handle = self.create_camera(i)
            self.cameras["front"].append(
                gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.env[i], self.camera_handle, front_camera
                    )
                )
            )

    def create_camera(self, env_idx):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height, camera_props.width = self.cfg.resolution
        camera_props.horizontal_fov = 40.0
        camera = self.gym.create_camera_sensor(self.env[env_idx], camera_props)

        camera_pos = gymapi.Vec3(1.7, 0.0, 0.9)
        camera_target = gymapi.Vec3(0.0, 0.0, 0.325)

        self.gym.set_camera_location(
            camera, self.env[env_idx], camera_pos, camera_target
        )
        return camera

    def default_asset_options(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        return asset_options

    def default_sim_params(self):
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / self.cfg.hz
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True if self.device == "cuda" else False

        sim_params.physx.use_gpu = True if self.device == "cuda" else False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.bounce_threshold_velocity = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.num_threads = 0
        return sim_params

    def refresh(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, False)
        self.gym.step_graphics(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
