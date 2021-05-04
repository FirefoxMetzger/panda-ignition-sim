import pandas as pd
from pathlib import Path
import random
import pickle

import generators

num_environments = 10
num_random_trajectories = 1
num_trajectories_per_goal = 10

dataset_root = Path(__file__).parents[0] / "dataset"
env_file_name = "environment.pkl"

env_metadata = pd.DataFrame(
    columns=[
        "NumGoals",
        "RandomTrajectories",
        "TrajectoriesPerGoal",
        "DataDir",
        "EnvironmentFilePath",
    ]
)

for env_idx in range(num_environments):
    environment_root = dataset_root / Path(f"environment_{env_idx}")
    environment_root.mkdir(exist_ok=True)

    num_goals = random.randint(4, 7)
    env = generators.generate_environment(num_goals)
    env_location = environment_root / "environment.pkl"
    with open(env_location, "wb") as file:
        pickle.dump(env, file)

    env_metadata.loc[env_idx] = [
        num_goals,
        num_random_trajectories,
        num_trajectories_per_goal,
        (environment_root).as_posix(),
        (environment_root / "environment.pkl").as_posix(),
    ]

env_metadata.loc[0, "TrajectoriesPerGoal"] = 100

env_metadata.to_excel(dataset_root / "environment_metadata.xlsx")

trajectory_metadata = pd.DataFrame(
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
        "worldTrajectoryRow"
    ]
)
trajectory_metadata.to_excel(dataset_root / "trajectory_metadata.xlsx")
