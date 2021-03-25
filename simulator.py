from scenario import gazebo as scenario_gazebo
from scenario import core as scenario_core
import gym_ignition_environments
from gym_ignition.rbd import conversions
from gym_ignition.context.gazebo import controllers


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
import ropy.ignition as ign
import ropy.transform as tf
import ropy.trajectory as rtj
from parsers import camera_parser, clock_parser

from panda_controller import LinearJointSpacePlanner

import time

gazebo = scenario_gazebo.GazeboSimulator(step_size=0.001, rtf=1.0, steps_per_run=1)
assert gazebo.insert_world_from_sdf("./panda_world_expanded.sdf")
gazebo.initialize()

# ---- initialize panda ----
# spawn
world = gazebo.get_world()
panda = gym_ignition_environments.models.panda.Panda(
    world=world, position=[0.2, 0, 1.025]
)
end_effector = panda.get_link("end_effector_frame")
gazebo.gui()
gazebo.run(paused=True)  # needs to be called before controller initialization

# Insert the ComputedTorqueFixedBase controller
assert panda.to_gazebo().insert_model_plugin(
    *controllers.ComputedTorqueFixedBase(
        kp=[100.0] * (panda.dofs() - 2) + [10000.0] * 2,
        ki=[0.0] * panda.dofs(),
        kd=[17.5] * (panda.dofs() - 2) + [100.0] * 2,
        urdf=panda.get_model_file(),
        joints=panda.joint_names(),
    ).args()
)

assert panda.set_joint_position_targets(panda.joint_positions())
assert panda.set_joint_velocity_targets(panda.joint_velocities())
assert panda.set_joint_acceleration_targets(panda.joint_accelerations())
home_position = np.array(end_effector.position())

panda_ctrl = LinearJointSpacePlanner(panda, control_frequency=gazebo.step_size())

ik_joints = [
    j.name() for j in panda.joints() if j.type is not scenario_core.JointType_fixed
]

# --- end initialize panda ---


# --- initialize camera projection ---
camera = gazebo.get_world().get_model("camera").get_link("link")
camera_frequency = 30  # Hz
steps_per_frame = round((1 / gazebo.step_size()) / camera_frequency)

# intrinsic matrix
cam_intrinsic = tf.projections.camera_frustum(hfov=1.047, image_shape=(1080, 1920))

# extrinsic matrix
cam_pos_world = np.array(camera.position())
cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
cam_extrinsic = tf.coordinates.transform(camera_frame_world)
# --- end initialize camera projection ---

# --- begin define placeable area ---
center = world.get_model("table2").get_link("link").position() + np.array((0, 0, 1.015))
# --- end define placeable area ---


fig, ax = plt.subplots(1)
with ign.Subscriber("/clock", parser=clock_parser) as clock_topic, ign.Subscriber(
    "/camera", parser=camera_parser
) as camera_topic:
    # Fix: the first step doesn't generate messages.
    # I don't exactly know why; I assume it has
    # to do with subscriptions being updated at the end
    # of the sim loop instead of the beginning?
    gazebo.run(paused=True)

    total_steps = 20000
    for sim_step in range(total_steps):
        # get the current time (published each step)
        sim_time = clock_topic.recv()

        # get synced camera images (published at 30Hz)
        if round(sim_time / gazebo.step_size()) % steps_per_frame == 0:
            img_msg = camera_topic.recv()

            # get px coordinates of endeffector
            eff_world = tf.homogenize(end_effector.position())
            eff_cam = np.matmul(cam_extrinsic, eff_world)
            eff_px_hom = np.matmul(cam_intrinsic, eff_cam)
            eff_px = tf.cartesianize(eff_px_hom)

            ax.add_patch(Circle(eff_px, radius=6))

            # verify that image message and simulator are in sync
            assert sim_time == img_msg.time

        if sim_step % 1000 == 0 and sim_step > 0:
            # get target on screen
            in_world = tf.homogenize(world_target)
            in_cam = np.matmul(cam_extrinsic, in_world)
            in_px_hom = np.matmul(cam_intrinsic, in_cam)
            in_px = tf.cartesianize(in_px_hom)
            ax.add_patch(Circle(in_px, radius=10, color="red"))

            ax.imshow(img_msg.image)
            ax.set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.show()
            fig, ax = plt.subplots(1)

        if sim_step % 2000 == 0:
            world_target = center + np.array((0, 0, 0.05))
            joint_target = panda_ctrl.solve_ik(world_target)
            t = np.arange(sim_step + 1, sim_step + 1001)
            move_to_goal = panda_ctrl.plan(
                t,
                [home_position, world_target],
                t_begin=sim_step,
                t_end=sim_step + 1000,
            )
            t = np.arange(sim_step + 1001, sim_step + 2001)
            move_from_goal = panda_ctrl.plan(
                t,
                [world_target, home_position],
                t_begin=sim_step + 1000,
                t_end=sim_step + 2000,
            )
            full_trajectory = np.empty((2000, 9))
            full_trajectory[:1000] = move_to_goal
            full_trajectory[1000:] = move_from_goal
            # full_trajectory[:, :] = move_to_goal[None, -1]
            plan_step = sim_step

        panda.set_joint_position_targets(
            full_trajectory[sim_step - plan_step], ik_joints
        )

        gazebo.run()

gazebo.close()
