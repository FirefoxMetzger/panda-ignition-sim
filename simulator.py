from ign_subscriber import IgnSubscriber
from parsers import camera_parser, clock_parser

from scenario import gazebo as scenario_gazebo
from gym_ignition.context.gazebo import controllers
import gym_ignition_environments

import numpy as np
import matplotlib.pyplot as plt

import time

# scenario_gazebo.set_verbosity(scenario_gazebo.Verbosity_info)

gazebo = scenario_gazebo.GazeboSimulator(
    step_size=0.001,
    rtf=1.0,
    steps_per_run=1
)
assert gazebo.insert_world_from_sdf("./panda_world_expanded.sdf")
gazebo.initialize()

# spawn panda
world = gazebo.get_world()
panda = gym_ignition_environments.models.panda.Panda(
    world=world, position=[0.2, 0, 1.025])
assert panda.set_controller_period(period=gazebo.step_size())

panda.get_joint(joint_name="panda_finger_joint1").to_gazebo().set_max_generalized_force(max_force=500.0)
panda.get_joint(joint_name="panda_finger_joint2").to_gazebo().set_max_generalized_force(max_force=500.0)
# Insert the ComputedTorqueFixedBase controller
assert panda.to_gazebo().insert_model_plugin(*controllers.ComputedTorqueFixedBase(
    kp=[100.0] * (panda.dofs() - 2) + [10000.0] * 2,
    ki=[0.0] * panda.dofs(),
    kd=[17.5] * (panda.dofs() - 2) + [100.0] * 2,
    urdf=panda.get_model_file(),
    joints=panda.joint_names(),
).args())

assert panda.set_joint_position_targets(panda.joint_positions())
assert panda.set_joint_velocity_targets(panda.joint_velocities())
assert panda.set_joint_acceleration_targets(panda.joint_accelerations())

gazebo.gui()

camera_frequency = 30 # Hz
steps_per_frame = round((1/gazebo.step_size())/camera_frequency)

# Fix: available topics seem to only be updated at the end
# of a step. This allows the subscribers to find the topic's
# address
gazebo.run(paused=True)

with IgnSubscriber("/clock", parser=clock_parser) as clock_topic, \
     IgnSubscriber("/camera", parser=camera_parser) as camera_topic:
    # Fix: the first step doesn't generate messages.
    # I don't exactly know why; I assume it has
    # to do with subscriptions being updated at the end
    # of the sim loop instead of the beginning?
    gazebo.run(paused=True)

    # execute a bunch of steps (proof of concept)
    for sim_step in range(1000):
        gazebo.run()

        # get the current time (published each step)
        sim_time = clock_topic.recv()

        # get synced camera images (published at 30Hz)
        if (int(round(sim_time/gazebo.step_size()))) % steps_per_frame == 0:
            img_msg = camera_topic.recv()
            # plt.imshow(img_msg.image)
            # plt.savefig(f"frames/frame{sim_step:0>4d}.jpg")

            # ensure image message and simulator are in sync
            assert sim_time == img_msg.time

gazebo.close()
