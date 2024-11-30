from dataclasses import dataclass, asdict, field


@dataclass
class ability_config(object):
    device: str = "cpu"
    collision: bool = False
    debug: bool = False
    env_spacing: float = 1.25
    up_axis: str = "Z"
    num_envs: int = 1
    hz: float = 60.0
    substeps: int = 4
    num_position_iterations: int = 8
    num_velocity_iterations: int = 2
    fix_base_link: bool = True
    num_dof: int = 34
    urdf_file: str = "xarm7_ability/xarm7_ability.urdf"

    # total_init: list = [
    #     -1.5899987,
    #     -0.3199993,
    #     -0.05,
    #     0.45,
    #     3.1599,
    #     0.78999954,
    #     -0.01,
    #     0.26121926,
    #     1.000002,
    #     0.21003355,
    #     0.9458213,
    #     0.4919993,
    #     1.2442857,
    #     0.3111902,
    #     1.0528969,
    #     -0.17063709,
    #     0.83448,
    #     1.5899987,
    #     -0.3199993,
    #     -0.05,
    #     0.45,
    #     3.1599,
    #     0.78999954,
    #     -0.01,
    #     0.42416182,
    #     1.1724788,
    #     0.42094582,
    #     1.1690747,
    #     0.58770615,
    #     1.3455927,
    #     0.48475993,
    #     1.2366228,
    #     -0.02744374,
    #     0.59406483,
    # ]
    # left_arm_init: list = [
    #     -1.5899987,
    #     -0.3199993,
    #     -0.05,
    #     0.45,
    #     3.1599,
    #     0.78999954,
    #     -0.01,
    # ]
    # right_arm_init: list = [
    #     1.5899987,
    #     -0.3199993,
    #     -0.05,
    #     0.45,
    #     3.1599,
    #     0.78999954,
    #     -0.01,
    # ]

    def to_dict(self):
        return asdict(self)


@dataclass
class h1_config(object):
    device: str = "cpu"
    collision: bool = False
    debug: bool = False
    env_spacing: float = 1.25
    up_axis: str = "Z"
    num_envs: int = 1
    hz: float = 60.0
    substeps: int = 4
    num_position_iterations: int = 8
    num_velocity_iterations: int = 2
    fix_base_link: bool = True
    num_dof: int = 51
    urdf_file: str = "h1_inspire/urdf/h1_inspire.urdf"

    def to_dict(self):
        return asdict(self)


CONFIG_DICT = {
    "ability": ability_config(),
    "h1": h1_config(),
}
