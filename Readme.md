# Simulation Environment for Legibility with Panda

## Installation

1. Install `gym-ignition` following the [official instructions](https://github.com/robotology/gym-ignition#setup)
2. `pip install pyzmq numpy matplotlib`
3. Install [ropy](https://github.com/FirefoxMetzger/ropy)

## Scripts

- `sim_runner.py env_idx goal_idx trajectory_idx` A script that sets up the
  environment with ID `env_idx`, samples a single trajectory to goal `goal_idx`,
  and records a 2d visualization of the trajectory, a video of the trajectories
  execution as well as numpy arrays of the trajectory in joint, world, and pixel
  space.
- `scaffold_dataset.py` A script to set up the folder structure of the dataset
  and to generate the environments
- `create_trajectory_dataset.py` A script to populate the dataset with the
  actual simulation. The script is aware of current progress, will continue
  where it left off if interrupted, and will skip existing runs. Essentially it
  calls `sim_runner.py` in a coordinated fashion.
- `add_observer_scores.py` A script that computes the attributed legibility
  scores for each behavior.