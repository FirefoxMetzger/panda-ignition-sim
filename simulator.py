from scenario import gazebo as scenario_gazebo
from scenario import core as scenario_core
from gym_ignition.context.gazebo import controllers
import gym_ignition_environments
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from gym_ignition.rbd import conversions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
import ropy.ignition as ign
import ropy.transform as tf
from parsers import camera_info_parser, camera_parser, clock_parser

import time

# scenario_gazebo.set_verbosity(scenario_gazebo.Verbosity_info)

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

# controllers
assert panda.set_controller_period(period=gazebo.step_size())
panda.get_joint(joint_name="panda_finger_joint1").to_gazebo().set_max_generalized_force(
    max_force=500.0
)
panda.get_joint(joint_name="panda_finger_joint2").to_gazebo().set_max_generalized_force(
    max_force=500.0
)
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

# Inverse Kinematics
ik_joints = [
    j.name() for j in panda.joints() if j.type is not scenario_core.JointType_fixed
]
ik = inverse_kinematics_nlp.InverseKinematicsNLP(
    urdf_filename=panda.get_model_file(),
    considered_joints=ik_joints,
    joint_serialization=panda.joint_names(),
)

ik.initialize(
    verbosity=1,
    floating_base=False,
    cost_tolerance=1e-8,
    constraints_tolerance=1e-8,
    base_frame=panda.base_frame(),
)

ik.set_current_robot_configuration(
    base_position=np.array(panda.base_position()),
    base_quaternion=np.array(panda.base_orientation()),
    joint_configuration=np.array(panda.joint_positions()),
)

ik.add_target(
    frame_name="end_effector_frame",
    target_type=inverse_kinematics_nlp.TargetType.POSE,
    as_constraint=False,
)


def solve_ik(target_position: np.ndarray) -> np.ndarray:

    quat_xyzw = R.from_euler(seq="y", angles=90, degrees=True).as_quat()

    ik.update_transform_target(
        target_name=ik.get_active_target_names()[0],
        position=target_position,
        quaternion=conversions.Quaternion.to_wxyz(xyzw=quat_xyzw),
    )

    # Run the IK
    ik.solve()

    return ik.get_reduced_solution().joint_configuration


def random_point_in_workspace():
    # sometimes not in workspace if it picks the edges
    target_position = np.random.uniform(
        low=[0.45, -0.7, 1.015], high=[0.45, 0.7, 1.015]
    ) + np.array([0, 0, 0.05])
    joint_target = solve_ik(target_position)
    return joint_target, target_position


# --- end initialize panda ---


# --- initialize trajectory ---
keyframes = np.load("initial_trajectory.npy").reshape((-1, 7))
direction = np.vstack((keyframes, keyframes[-1, ...][None, ...]))[1:, ...] - keyframes
num_keyframes = keyframes.shape[0]


def desired_position(t):
    """
    Return the desired position along the trajectory at time t in [0, 1]

    This assumes a simple linear velocity profile
    """
    if t > 1:
        return keyframes[-1, ...]

    # progress in keyframe list
    idx = np.floor(t * (num_keyframes - 1)).astype(np.int64)
    tween = (t * num_keyframes) % 1  # progress between two keyframes
    return direction[idx, ...] * tween + keyframes[idx, ...]


# --- end initialize trajectory ---


# --- initialize camera projection ---
camera = gazebo.get_world().get_model("camera").get_link("link")

# intrinsic matrix
cam_intrinsic = tf.projections.camera_frustum(hfov=1.047, image_shape=(1080, 1920))

# extrinsic matrix
cam_pos_world = np.array(camera.position())
cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
cam_extrinsic = tf.coordinates.transform(camera_frame_world)
# --- end initialize camera projection ---


gazebo.gui()

camera_frequency = 30  # Hz
steps_per_frame = round((1 / gazebo.step_size()) / camera_frequency)

# Fix: available topics seem to only be updated at the end
# of a step. This allows the subscribers to find the topic's
# address
gazebo.run(paused=True)

# wait a second for all async processes to startup
time.sleep(1)

fig, ax = plt.subplots(1)
with ign.Subscriber("/clock", parser=clock_parser) as clock_topic, ign.Subscriber(
    "/camera", parser=camera_parser
) as camera_topic:
    # Fix: the first step doesn't generate messages.
    # I don't exactly know why; I assume it has
    # to do with subscriptions being updated at the end
    # of the sim loop instead of the beginning?
    gazebo.run(paused=True)
    home_position = np.array(panda.joint_positions())

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
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.show()
            fig, ax = plt.subplots(1)

        if sim_step % 2000 == 0:
            joint_target, world_target = random_point_in_workspace()


        alpha = (sim_step % 1000) / 1000
        if sim_step % 2000 < 1000:
            # go to target
            current_point = alpha * joint_target + (1 - alpha) * home_position
        else:
            # go home
            current_point = alpha * home_position + (1 - alpha) * joint_target

        # update controllers
        # panda.set_joint_position_targets([*pos.tolist(), 1.0, 1.0], ik_joints)
        panda.set_joint_position_targets(current_point, ik_joints)

        gazebo.run()

gazebo.close()
