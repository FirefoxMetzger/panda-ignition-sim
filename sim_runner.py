import imageio as iio
import matplotlib.pyplot as plt
import ropy.ignition as ign
import sys
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import random
from datetime import datetime

import generators
from simulator import LegibilitySimulator
from matplotlib.patches import Circle
from parsers import camera_parser
from scenario import gazebo as scenario_gazebo


env_idx = int(sys.argv[1])
goal_idx = int(sys.argv[2])
trajectory_idx = int(sys.argv[3])
random.seed(datetime.now())

dataset_root = Path(__file__).parents[0] / "dataset"
env_meta = pd.read_excel(dataset_root / "environment_metadata.xlsx").set_index("Unnamed: 0")
trajectory_meta = pd.read_excel(dataset_root / "trajectory_metadata.xlsx").set_index("Unnamed: 0")
env_root = Path(".") / env_meta.loc[env_idx].DataDir

with open(env_meta.loc[env_idx].EnvironmentFilePath, "rb") as file:
    env = pickle.load(file)

trajectory_row = pd.DataFrame(
    [[
        env_idx,
        False,
        goal_idx,
        trajectory_idx,
        random.randint(0, len(env.control_points)),
        env_root / f"goal_{goal_idx}_trajectory_{trajectory_idx}.mp4",
        env_root / f"goal_{goal_idx}_trajectory_{trajectory_idx}.png",
        env_root / f"goal_{goal_idx}_trajectory_{trajectory_idx}_camera_trajectory.npy",
        0,
        env_root / f"goal_{goal_idx}_trajectory_{trajectory_idx}_joint_trajectory.npy",
        0,
        env_root / f"goal_{goal_idx}_trajectory_{trajectory_idx}_world_trajectory.npy",
        0,
    ]],
    columns=[
        "environmentID",
        "isRandom",
        "targetGoal",
        "trajectory_idx",
        "viaPointIdx",
        "videoFile",
        "imageFile",
        "cameraTrajectoryFile",
        "cameraTrajectoryRow",
        "jointTrajectoryFile",
        "jointTrajectoryRow",
        "worldTrajectoryFile",
        "worldTrajectoryRow",
    ],
)

# execute the trajectory and record
# - endeffector position in planning space

cam_trajectory = list()
joint_trajectory = list()
world_trajectory = list()
writer = iio.get_writer(trajectory_row.iloc[0]["videoFile"], format="FFMPEG", mode="I", fps=30)
fig, ax = plt.subplots(1)
panda_config = {"position": [0.2, 0, 1.025]}
with LegibilitySimulator(
    panda_config=panda_config,
    environment=env,
    step_size=0.001,
    rtf=1.0,
    steps_per_run=round((1 / 0.001) / 30),
) as simulator:
    goal_px = simulator.in_px_coordinates(simulator.cubes[goal_idx].base_position())
    ax.add_patch(Circle(goal_px, radius=10, color="red"))

    cubes_world = np.stack([cube.base_position() for cube in simulator.cubes])
    np.save(env_root / "world_cube_position.npy", cubes_world)
    cubes_cam = np.stack([simulator.in_px_coordinates(pos) for pos in cubes_world])
    np.save(env_root / "camera_cube_position.npy", cubes_cam)
    cubes_joint = np.stack([simulator.panda.solve_ik(position=pos) for pos in cubes_world])
    np.save(env_root / "joint_cube_position.npy", cubes_joint)

    # uncomment to show the GUI
    # simulator.gui()
    simulator.prepare_goal_trajectory(goal_idx, via_point_idx=trajectory_row.iloc[0]["viaPointIdx"])
    with ign.Subscriber("/camera", parser=camera_parser) as camera_topic:
        simulator.run(paused=True)
        for sim_step in range(330):
            img_msg = camera_topic.recv()
            writer.append_data(img_msg.image)

            eff_px = simulator.in_px_coordinates(simulator.panda.tool_pose[0])

            cam_trajectory.append(eff_px)
            joint_trajectory.append(simulator.panda.position)
            world_trajectory.append(simulator.panda.tool_pose[0])

            ax.add_patch(Circle(eff_px, radius=5))
            simulator.run()

writer.close()

np.save(
    trajectory_row.iloc[0]["cameraTrajectoryFile"],
    np.stack(cam_trajectory, axis=0)[None, ...],
)
np.save(
    trajectory_row.iloc[0]["worldTrajectoryFile"],
    np.stack(world_trajectory, axis=0)[None, ...],
)
np.save(
    trajectory_row.iloc[0]["jointTrajectoryFile"],
    np.stack(joint_trajectory, axis=0)[None, ...],
)

# visualize the trajectory
ax.imshow(img_msg.image)
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
fig.savefig(trajectory_row.iloc[0]["imageFile"])

trajectory_meta = trajectory_meta.append(trajectory_row)
trajectory_meta.to_excel(dataset_root / "trajectory_metadata.xlsx")
