from scenario import gazebo as scenario_gazebo
from scenario import core as scenario_core
import gym_ignition_environments
from gym_ignition.rbd import conversions
from gym_ignition.context.gazebo import controllers
import gym_ignition

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
import ropy.ignition as ign
import ropy.transform as rtf
import ropy.trajectory as rtj
from parsers import camera_parser, clock_parser
from dataclasses import dataclass

from panda_controller import LinearJointSpacePlanner
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

        if isinstance(velocity, np.array):
            velocity = velocity.tolist()

        if isinstance(angular_velocity, np.array):
            angular_velocity = angular_velocity.tolist()

        model = self.model_cache[model_template].format(**template_parameters)
        model_name = gym_ignition.utils.scenario.get_unique_model_name(
            world=self.world, model_name=name_prefix
        )
        pose = scenario_core.Pose(position, orientation)
        assert self.world.insert_model(model, pose, model_name)

        obj = world.get_model(model_name)

        velocity = scenario_core.Array3d(velocity.tolist())
        assert cube.to_gazebo().reset_base_world_linear_velocity(velocity)

        angular_velocity = scenario_core.Array3d(angular_velocity.tolist())
        assert cube.to_gazebo().reset_base_world_angular_velocity(velocity)

        return obj


class LegibilitySimulator(ModelSpawnerMixin, PandaMixin, scenario_gazebo.GazeboSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_callbacks = list()

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

        try:

            world = self.world

            # --- initialize camera projection ---
            camera = self.get_world().get_model("camera").get_link("link")
            camera_frequency = 30  # Hz
            steps_per_frame = round((1 / self.step_size()) / camera_frequency)

            # intrinsic matrix
            cam_intrinsic = rtf.perspective_frustum(
                hfov=1.13446, image_shape=(1080, 1920)
            )

            # extrinsic matrix
            cam_pos_world = np.array(camera.position())
            cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
            cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
            camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
            cam_extrinsic = rtf.transform(camera_frame_world)
            # --- end initialize camera projection ---

            num_cubes = 6
            env = generators.generate_environment(num_cubes)
            cubes = list()
            for cube in env.cubes:
                cubes.append(self.insert_model())
        except Exception:
            self.close()
            raise RuntimeError("Failed to setup simulation context.")

    def simulate(self, *, pre_step_callbacks=None, post_step_callbacks=None):
        if pre_step_callbacks is None:
            pre_step_callbacks = list()
        if post_step_callbacks is None:
            post_step_callbacks = list()

        with ign.Subscriber(
            "/clock", parser=clock_parser
        ) as clock_topic, ign.Subscriber(
            "/camera", parser=camera_parser
        ) as camera_topic:
            self.run(paused=True)

            num_targets = 1
            trajectory_duration = 5000
            total_steps = trajectory_duration * num_targets
            for sim_step in range(total_steps):
                # get the current time (published each step)
                sim_time = clock_topic.recv()

                # get synced camera images (published at 30Hz)
                if round(sim_time / self.step_size()) % steps_per_frame == 0:
                    img_msg = camera_topic.recv()
                    writer.append_data(img_msg.image)

                    # get px coordinates of endeffector
                    eff_world = rtf.homogenize(end_effector.position())
                    eff_cam = np.matmul(cam_extrinsic, eff_world)
                    eff_px_hom = np.matmul(cam_intrinsic, eff_cam)
                    eff_px = rtf.cartesianize(eff_px_hom)

                    ax.add_patch(Circle(eff_px, radius=6))

                    # verify that image message and simulator are in sync
                    assert sim_time == img_msg.time

                if sim_step == 0 or np.allclose(
                    end_effector.position(), world_target, atol=0.01
                ):
                    # reset the robot
                    panda.to_gazebo().reset_joint_positions(home_position_joints)
                    panda.to_gazebo().reset_joint_velocities(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    )
                    panda.set_joint_position_targets(home_position_joints)
                    panda.set_joint_velocity_targets([0, 0, 0, 0, 0, 0, 0, 0, 0])

                    cube_idx = random.randint(0, num_cubes - 1)
                    cube = cubes[cube_idx]
                    pos = np.array(cube.base_position())
                    ori = R.from_quat(np.array(cube.base_orientation())[[1, 2, 3, 0]])

                    world_target = pos + np.array((0, 0, 0.055))
                    pos_trajectory = generators.sample_trajectory(
                        home_position, world_target, env, num_control=2
                    )
                    pos_trajectory[1:-1, :] += panda.base_position()

                    world_ori = R.from_euler(seq="y", angles=(np.pi / 2))  # * ori
                    world_ori = world_ori.as_quat()[[3, 0, 1, 2]]
                    ori_trajectory = (home_orientation, world_ori)

                    t = np.arange(sim_step + 1, sim_step + trajectory_duration + 1)
                    trajectory = panda_ctrl.plan(
                        t,
                        [pos_trajectory, ori_trajectory],
                        t_begin=sim_step,
                        t_end=sim_step + trajectory_duration,
                    )
                    full_trajectory = trajectory
                    plan_step = sim_step

                    # visualize target
                    in_world = rtf.homogenize(world_target)
                    in_cam = np.matmul(cam_extrinsic, in_world)
                    in_px_hom = np.matmul(cam_intrinsic, in_cam)
                    in_px = rtf.cartesianize(in_px_hom)
                    ax.add_patch(Circle(in_px, radius=10, color="red"))

                time = min(trajectory_duration - 1, sim_step - plan_step)

                panda.set_joint_position_targets(full_trajectory[time, :, 0], ik_joints)
                panda.set_joint_velocity_targets(full_trajectory[time, :, 1], ik_joints)

                for callback in pre_step_callbacks:
                    callback(self)

                self.run()

                for callback in post_step_callbacks:
                    callback(self)

    def __exit__(self, type, value, traceback):
        self.close()




if __name__ == "__main__":
    foo = scenario_gazebo.GazeboSimulator(step_size=0.001, rtf=1.0, steps_per_run=1)
    foo.initialize()
    world = foo.get_world()
    bar = Panda(world=world, position=[0,0,0])
    import pdb; pdb.set_trace()
    # fig, ax = plt.subplots(1)

    # writer = iio.get_writer("my_video.mp4", format="FFMPEG", mode="I", fps=30)


    # with LegibilitySimulator(step_size=0.001, rtf=1.0, steps_per_run=1) as simulator:
    #     simulator.run()

    # # visualize the trajectory
    # ax.imshow(img_msg.image)
    # ax.set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
