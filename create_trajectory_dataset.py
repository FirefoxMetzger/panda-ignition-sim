import pandas as pd
import numpy as np
from pathlib import Path

from subprocess import call

num_random_trajectories = 50
num_trajectories_per_goal = 1



dataset_root = Path(__file__).parents[0] / "dataset"
dataset_root.mkdir(exist_ok=True)

env_metadata = pd.read_excel(dataset_root / "environment_metadata.xlsx").set_index("Unnamed: 0")
trajectory_metadata = pd.read_excel(dataset_root / "trajectory_metadata.xlsx").set_index("Unnamed: 0")


for env_idx, row in env_metadata.iterrows():
    (
        num_goals,
        num_random_trajectories,
        num_trajectories_per_goal,
        env_data_dir,
        env_path,
    ) = row

    env_trajectories = trajectory_metadata[trajectory_metadata.environmentID == env_idx]
    

    # load environment
    for goal_idx in range(num_goals):
        goal_trajectories = env_trajectories[env_trajectories.targetGoal == goal_idx]
        # initialize a new numpy array to store the planning-space trajectory
        for trajectory_index in range(num_trajectories_per_goal):
            if trajectory_index in goal_trajectories.trajectory_idx.values:
                print(f"--- SKIPPING env {env_idx}, goal {goal_idx}, traj {trajectory_index}")
                continue
            # save the trajectory into a numpy array

            # execute the trajectory and record
            # - frames (video)
            # - endeffector position in camera/world/state/planning space
            call(["python3", "sim_runner.py", str(env_idx), str(goal_idx), str(trajectory_index)])
            print(f"--- Done with env {env_idx}, goal {goal_idx}, traj {trajectory_index}")
            

    random_trajectories = env_trajectories[env_trajectories.isRandom == True]
    for trajectory_idx in range(num_random_trajectories):
        pass
        # reset the environment




# refactor simulation data
env_metadata = pd.read_excel(dataset_root / "environment_metadata.xlsx").set_index("Unnamed: 0")
trajectory_metadata = pd.read_excel(dataset_root / "trajectory_metadata.xlsx")

for prefix, dims in [("camera", 2), ("joint", 9), ("world", 3)]:
    data = np.empty((len(trajectory_metadata),330,dims), dtype=np.float_)
    file_loc = prefix + "TrajectoryFile"
    row_loc = prefix + "TrajectoryRow"

    for idx in range(len(trajectory_metadata)):
        file = trajectory_metadata.loc[idx, file_loc]
        pos = trajectory_metadata.loc[idx, row_loc]
        data[idx] = np.load(file)[pos]

    trajectory_location = dataset_root / (prefix+"_trajectory_data.npy")
    np.save(trajectory_location, data)

    for idx in range(len(trajectory_metadata)):
        trajectory_metadata.at[idx, file_loc] = trajectory_location
        trajectory_metadata.at[idx, row_loc] = idx

trajectory_metadata.to_excel(dataset_root / "trajectory_metadata.xlsx")