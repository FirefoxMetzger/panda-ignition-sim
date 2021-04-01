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

from panda_controller import LinearJointSpacePlanner
import imageio as iio
import generators

import time
import random

gazebo = scenario_gazebo.GazeboSimulator(step_size=0.001, rtf=1.0, steps_per_run=1)
assert gazebo.insert_world_from_sdf("./sdf/environment.sdf")
gazebo.initialize()
world = gazebo.get_world()

# ---- initialize panda ----
# spawn
panda = gym_ignition_environments.models.panda.Panda(
    world=world, position=[0.2, 0, 1.025]
)
panda.to_gazebo().enable_self_collisions(True)
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
home_orientation = np.array(end_effector.orientation())
home_pose = np.hstack((home_position, home_orientation[[1, 2, 3, 0]]))
home_position_joints = panda.joint_positions()

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
cam_intrinsic = rtf.perspective_frustum(hfov=1.13446, image_shape=(1080, 1920))

# extrinsic matrix
cam_pos_world = np.array(camera.position())
cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
cam_extrinsic = rtf.transform(camera_frame_world)
# --- end initialize camera projection ---

# --- initialize cube spawner ---
with open("./sdf/cube_template.sdf.in", "r") as file:
    cube_template = file.read()

def spawn_cube(cube):
    model = cube_template.format(diffuse=cube.color, ambient=cube.color)
    model_name = gym_ignition.utils.scenario.get_unique_model_name(
        world=world, model_name=cube.name
    )

    pos = cube.position + panda.base_position()
    assert world.insert_model(
        model, scenario_core.Pose(pos, cube.orientation), model_name
    )

    return world.get_model(model_name)
# --- end initialize cube spawner ---

# record video of the simulation
writer = iio.get_writer("my_video.mp4", format="FFMPEG", mode="I", fps=30)

time.sleep(3)

fig, ax = plt.subplots(1)

num_cubes = 6
env = generators.generate_environment(num_cubes)
cubes = list()
for cube in env.cubes:
    cubes.append(spawn_cube(cube))

with ign.Subscriber("/clock", parser=clock_parser) as clock_topic, ign.Subscriber(
    "/camera", parser=camera_parser
) as camera_topic:
    gazebo.run(paused=True)

    num_targets = 20
    total_steps = 2000 * num_targets
    for sim_step in range(total_steps):
        # get the current time (published each step)
        sim_time = clock_topic.recv()

        # get synced camera images (published at 30Hz)
        if round(sim_time / gazebo.step_size()) % steps_per_frame == 0:
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

        if sim_step % 2000 == 0:
            # reset the robot
            panda.to_gazebo().reset_joint_positions(home_position_joints)
            panda.to_gazebo().reset_joint_velocities([0,0,0,0,0,0,0,0,0])

            cube_idx = random.randint(0, num_cubes-1)
            cube = cubes[cube_idx]
            pos = np.array(cube.base_position())
            ori = R.from_quat(np.array(cube.base_orientation())[[1, 2, 3, 0]])

            world_target = pos + np.array((0, 0, 0.055))
            pos_trajectory = generators.sample_trajectory(home_position, world_target, env, num_control=5)
            pos_trajectory[1:-1, :] += panda.base_position()
            # import pdb; pdb.set_trace()

            world_ori = R.from_euler(seq="y", angles=(np.pi/2)) #* ori
            world_ori = world_ori.as_quat()[[3, 0, 1, 2]]
            ori_trajectory = (home_orientation, world_ori)

            t = np.arange(sim_step + 1, sim_step + 2001)
            move_to_goal = panda_ctrl.plan(
                t,
                [pos_trajectory, ori_trajectory],
                t_begin=sim_step,
                t_end=sim_step + 2000,
            )
            full_trajectory = move_to_goal
            plan_step = sim_step

            # visualize target
            in_world = rtf.homogenize(world_target)
            in_cam = np.matmul(cam_extrinsic, in_world)
            in_px_hom = np.matmul(cam_intrinsic, in_cam)
            in_px = rtf.cartesianize(in_px_hom)
            ax.add_patch(Circle(in_px, radius=10, color="red"))

        panda.set_joint_position_targets(
            full_trajectory[sim_step - plan_step], ik_joints
        )
        # panda.set_joint_position_targets(
        #     full_trajectory[-1], ik_joints
        # )

        gazebo.run()

writer.close()

# visualize the trajectory
ax.imshow(img_msg.image)
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.show()

gazebo.close()
