import pandas as pd
import numpy as np
from pathlib import Path


def observer1(trajectory, goals):
    """Scores Trajectories based on the perceived distance to each goal
    
    The value is given as logits
    """
    pointer = goals.T[None, ...] - trajectory[..., None]
    distance = np.linalg.norm(pointer, axis=1)

    return distance


dataset_root = Path(__file__).parents[0] / "dataset"
dataset_root.mkdir(exist_ok=True)

(dataset_root / "ratings").mkdir(exist_ok=True)

env_metadata = pd.read_excel(dataset_root / "environment_metadata.xlsx").set_index("Unnamed: 0")
trajectory_metadata = pd.read_excel(dataset_root / "trajectory_metadata.xlsx")

observers = {
    "observer1": observer1
}

scores = {observer:list() for observer in observers}
for _, row in trajectory_metadata.iterrows():
    env_dir = Path(env_metadata.loc[row.environmentID, "DataDir"])
    goal_positions = np.load(env_dir / "camera_cube_position.npy")
    trajectory = np.load(row.cameraTrajectoryFile)[row.cameraTrajectoryRow]
    for name in observers:
        score = observers[name](trajectory, goal_positions)
        scores[name].append(score)

for name in observers:
    ratings = np.empty(len(scores[name]), dtype=object)
    ratings[:] = scores[name]
    np.save(dataset_root / "ratings" / (name+".npy"), ratings)
