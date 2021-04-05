import pandas as pd
import numpy as np

environment_seeds = list()
num_random_trajectories = 50
num_trajectories_per_goal = 50
via_points = 2
dataset_root_dir = ""


for environment in environment_seeds:
    # create a new folder for the environment
    # create a new pandas dataframe for structured data
    # add goal positions for all the spaces to dataframe
    # load environment
    for goal in environment.goals:
        # initialize new numpy array with shape (num_goals, control_points, DoF)
        # initialize a new numpy array to store the camera trajectory
        # initialize a new numpy array to store the world trajectory
        # initialize a new numpy array to store the state-space trajectory
        # initialize a new numpy array to store the planning-space trajectory
        for trajectory_index in range(num_trajectories_per_goal):
            # sample a trajectory using the specified number
            # of via points

            # save the trajectory into a numpy array

            # execute the trajectory and record
            # - frames (video)
            # - endeffector position in camera/world/state/planning space
            pass

        # reset the environment

    for trajectory_idx in range(num_random_trajectories):
        pass

        # reset the environment