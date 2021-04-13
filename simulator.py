from scenario import gazebo as scenario_gazebo
from scenario import core as scenario_core
import gym_ignition_environments
from gym_ignition.rbd import conversions
import gym_ignition
from gym_ignition.context.gazebo import controllers


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
import ropy.ignition as ign
import ropy.transform as rtf
import ropy.trajectory as rtj
from ropy.trajectory import spline_trajectory
from parsers import camera_parser, clock_parser
from dataclasses import dataclass

from panda_wrapper import PandaMixin
import imageio as iio
import generators

import time
import random

import numpy.typing as npt
from typing import Tuple


class ModelSpawnerMixin:
    """Simulator Mixin to spawn objects"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cache = dict()

    def insert_model(
        self,
        model_template: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        velocity: Tuple[float, float, float] = (0, 0, 0),
        angular_velocity: Tuple[float, float, float] = (0, 0, 0),
        *,
        name_prefix="",
        **template_parameters
    ):
        """Spawn the model into the simulation world"""

        if model_template not in self.model_cache:
            with open(model_template, "r") as file:
                self.model_cache[model_template] = file.read()

        if isinstance(velocity, np.ndarray):
            velocity = velocity.tolist()

        if isinstance(angular_velocity, np.ndarray):
            angular_velocity = angular_velocity.tolist()

        world = self.world

        model = self.model_cache[model_template].format(**template_parameters)
        model_name = gym_ignition.utils.scenario.get_unique_model_name(
            world=world, model_name=name_prefix
        )
        pose = scenario_core.Pose(position, orientation)
        assert world.insert_model(model, pose, model_name)

        obj = world.get_model(model_name)

        velocity = scenario_core.Array3d(velocity)
        assert obj.to_gazebo().reset_base_world_linear_velocity(velocity)

        angular_velocity = scenario_core.Array3d(angular_velocity)
        assert obj.to_gazebo().reset_base_world_angular_velocity(velocity)

        return obj


class CameraMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = None

        # --- initialize camera projection ---
        camera_frequency = 30  # Hz
        self.steps_per_frame = round((1 / self.step_size()) / camera_frequency)
        self.runs_per_frame = round(self.steps_per_run() / self.steps_per_frame)

        self.cam_intrinsic = rtf.perspective_frustum(
            hfov=1.13446, image_shape=(1080, 1920)
        )
        self.cam_extrinsic = None

    def initialize(self):
        super().initialize()
        self.camera = self.get_world().get_model("camera").get_link("link")

        # extrinsic matrix
        camera = self.camera
        cam_pos_world = np.array(camera.position())
        cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
        cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
        camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
        self.cam_extrinsic = rtf.transform(camera_frame_world)

    def in_px_coordinates(self, position):
        pos_world = rtf.homogenize(position)
        pos_cam = np.matmul(self.cam_extrinsic, pos_world)
        pos_px_hom = np.matmul(self.cam_intrinsic, pos_cam)
        pos_px = rtf.cartesianize(pos_px_hom)

        return pos_px


class LegibilitySimulator(
    ModelSpawnerMixin, CameraMixin, PandaMixin, scenario_gazebo.GazeboSimulator
):
    def __init__(self, environment: generators.Environment, **kwargs):
        super().__init__(**kwargs)

        self.env = environment
        self.cubes = list()

        self.path = None
        self.path_velocity = None
        self.path_time_start = 0
        self.path_time_end = 10


    @property
    def world(self):
        return self.get_world()

    def initialize(self):
        raise NotImplementedError(
            "Do not initialize the simulator manually. "
            "Use 'with LegibiltySimulator(...) as' instead."
        )

    def __enter__(self):
        assert self.insert_world_from_sdf("./sdf/environment.sdf")
        super().initialize()

        panda_base = self.panda.base_position()

        for cube in self.env.cubes:
            cube_item = self.insert_model(
                "./sdf/cube_template.sdf.in",
                cube.position + panda_base,
                name_prefix="cube",
                diffuse=cube.color,
                ambient=cube.color,
            )
            self.cubes.append(cube_item)

        return self

    def prepare_goal_trajectory(self, cube_idx):
        cube = self.cubes[cube_idx]
        cube_position = np.array(cube.base_position())
        ori = R.from_quat(np.array(cube.base_orientation())[[1, 2, 3, 0]])

        tool_pos, tool_rot = self.panda.tool_pose

        idx = random.randint(0, len(self.env.control_points))
        via_point = self.env.control_points[idx] + self.panda.base_position()

        # key in the movement's poses
        #   0 - home_position
        #   1 - random via-point
        #   2 - above cube
        #   3 - at cube (open gripper)
        #   4 - grabbing cube (closed gripper)
        #   5 - home_position (holding cube)
        pose_keys = np.empty((6, 9), dtype=np.float_)
        pose_keys[0] = self.panda.home_position
        pose_keys[1] = self.panda.solve_ik(position=via_point)
        pose_keys[2] = self.panda.solve_ik(position=(cube_position + np.array((0, 0, 0.01))))
        pose_keys[2, -2:] = (.04, .04)
        pose_keys[3] = self.panda.solve_ik(position=cube_position)
        pose_keys[3, -2:] = (.04, .04)
        pose_keys[4] = self.panda.solve_ik(position = cube_position)
        pose_keys[4, -2:] = (0, 0)
        pose_keys[5] = self.panda.home_position
        pose_keys[5, -2:] = (0, 0)


        # set keyframe times
        trajectory_duration = self.path_time_end - self.path_time_start
        times = np.array([0, 0.275, 0.55, 0.6, 0.7, 1]) * trajectory_duration + self.path_time_start

        if self.path_time_start % self.step_size() != 0:
            raise RuntimeError("Path does not start during a simulator step.")

        t = np.arange(
            self.path_time_start+self.step_size(),
            self.path_time_end+self.step_size(),
            self.step_size()
        )
        self.path = spline_trajectory(
            t, pose_keys, t_control=times, degree=1
        )
        self.path_velocity = spline_trajectory(
            t,
            pose_keys,
            t_control=times,
            degree=1,
            derivative=1,
        )


    def run(self, **kwargs):
        sim_time = self.world.time()
        if self.path_time_start < sim_time < self.path_time_end:
            idx = int(round((sim_time - self.path_time_start) / self.step_size()))
            self.panda.target_position = self.path[idx]
            self.panda.target_velocity = self.path_velocity[idx]

        super().run(**kwargs)


    def __exit__(self, type, value, traceback):
        assert self.close()


if __name__ == "__main__":
    fig, ax = plt.subplots(1)

    writer = iio.get_writer("my_video.mp4", format="FFMPEG", mode="I", fps=30)
    env = generators.generate_environment(6)

    panda_config = {"position": [0.2, 0, 1.025]}

    with LegibilitySimulator(
        panda_config=panda_config,
        environment=env,
        step_size=0.001,
        rtf=1.0,
        steps_per_run=round((1 / 0.001) / 30),
    ) as simulator:
        goal_px = simulator.in_px_coordinates(simulator.cubes[0].base_position())
        ax.add_patch(Circle(goal_px, radius=10, color="red"))

        # simulator.gui()
        simulator.prepare_goal_trajectory(0)
        with ign.Subscriber("/camera", parser=camera_parser) as camera_topic:
            simulator.run(paused=True)
            for sim_step in range(330):
                img_msg = camera_topic.recv()
                writer.append_data(img_msg.image)

                # get px coordinates of endeffector
                eff_px = simulator.in_px_coordinates(simulator.panda.tool_pose[0])
                ax.add_patch(Circle(eff_px, radius=5))
                simulator.run()

    writer.close()

    # visualize the trajectory
    ax.imshow(img_msg.image)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.show()
