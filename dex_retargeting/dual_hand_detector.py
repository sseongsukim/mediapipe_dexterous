import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class DualHandDector(object):

    def __init__(
        self,
        left_retargeting,
        right_retargeting,
        depth_scale,
        selfie=False,
    ):
        self.mpHands = mp.solutions.hands
        self.hand_detectors = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.left_retargeting = left_retargeting
        self.right_retargeting = right_retargeting
        self.inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.depth_scale = depth_scale
        if selfie:
            self.left_operator2mano = OPERATOR2MANO_RIGHT
            self.right_operator2mano = OPERATOR2MANO_LEFT
        else:
            self.left_operator2mano = OPERATOR2MANO_LEFT
            self.right_operator2mano = OPERATOR2MANO_RIGHT

    def get_info(self, hand_side, keypoint_3d):
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        operator2mano = (
            self.left_operator2mano if hand_side == "Left" else self.right_operator2mano
        )
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano

        retargeting = (
            self.left_retargeting if hand_side == "Left" else self.right_retargeting
        )

        indices = retargeting.optimizer.target_link_human_indices
        origin_indices, task_indices = indices[0, :], indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)
        return qpos, mediapipe_wrist_rot, joint_pos

    def detect(self, rgb, depth_image_flipped):
        results = self.hand_detectors.process(rgb)
        info = {}
        if results.multi_hand_landmarks:
            for i in range(len(results.multi_hand_landmarks)):
                self.mpDraw.draw_landmarks(
                    rgb,
                    results.multi_hand_landmarks[i],
                    self.mpHands.HAND_CONNECTIONS,
                )
                hand_side = results.multi_handedness[i].classification[0].label
                wrist_landmark = results.multi_hand_landmarks[i].landmark[0]
                x = int(wrist_landmark.x * len(depth_image_flipped[0]))
                y = int(wrist_landmark.y * len(depth_image_flipped))

                if x >= len(depth_image_flipped[0]):
                    x = len(depth_image_flipped[0]) - 1
                if y >= len(depth_image_flipped):
                    y = len(depth_image_flipped) - 1
                distance = depth_image_flipped[y, x] * self.depth_scale

                qpos, wrist_rot, joint_pos = self.get_info(
                    self.inverse_hand_dict[hand_side],
                    results.multi_hand_world_landmarks[i],
                )
                if hand_side == "Left":
                    wrist_xy = (
                        -results.multi_hand_landmarks[i].landmark[0].y,
                        -results.multi_hand_landmarks[i].landmark[0].x,
                    )
                elif hand_side == "Right":
                    wrist_xy = (
                        -results.multi_hand_landmarks[i].landmark[0].y,
                        -results.multi_hand_landmarks[i].landmark[0].x,
                    )

                wrist_pos = np.array(
                    [wrist_xy[0], wrist_xy[1], distance], dtype=np.float32
                )
                info[hand_side] = {
                    "qpos": qpos,
                    "wrist_rot": wrist_rot,
                    "wrist_pos": wrist_pos,
                    "joint_pos": joint_pos,
                }
        else:
            return None
        return info

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
