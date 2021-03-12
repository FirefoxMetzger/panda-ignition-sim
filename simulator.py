from ignition.msgs.image_pb2 import Image
from ignition.msgs.clock_pb2 import Clock
from scenario import gazebo as scenario_gazebo
import gym_ignition_environments
import numpy as np
import matplotlib.pyplot as plt

from ign_subscriber import IgnSubscriber

import time
import zmq

# scenario_gazebo.set_verbosity(scenario_gazebo.Verbosity_info)

step_size = 0.001
gazebo = scenario_gazebo.GazeboSimulator(
    step_size=step_size,
    rtf=1.0,
    steps_per_run=1
)
assert gazebo.insert_world_from_sdf("./panda_world_expanded.sdf")
gazebo.initialize()

world = gazebo.get_world()
panda = gym_ignition_environments.models.panda.Panda(
    world=world, position=[0.2, 0, 1.025])

gazebo.gui()

camera_frequency = 30 # Hz
steps_per_frame = round((1/step_size)/camera_frequency)

# Fix: available topics seem to only be updated at the end
# of a step. This allows the subscribers to find the topic's
# address
gazebo.run(paused=True)

with IgnSubscriber("/clock") as clock_topic, \
     IgnSubscriber("/camera") as camera_topic:
    # Fix: the first step doesn't generate messages.
    # I don't exactly know why; I assume it has
    # to do with subscriptions being updated at the end
    # of the sim loop instead of the beginning?
    gazebo.run(paused=True)

    # execute a bunch of steps (proof of concept)
    for sim_step in range(1000):
        gazebo.run()

        # sleep to await messages for testing
        # time.sleep(0.04)

        # get the current clock (published each step)
        zmq_msg = clock_topic.recv()
        clock_msg = Clock()
        clock_msg.ParseFromString(zmq_msg[2])
        sim_time = clock_msg.sim.sec + clock_msg.sim.nsec*1e-9

        # get synced camera images (published at 30Hz)
        if (int(round(sim_time/step_size))) % steps_per_frame == 0:
            zmq_msg = camera_topic.recv()
            image_msg = Image()
            image_msg.ParseFromString(zmq_msg[2])
            im = np.frombuffer(image_msg.data, dtype=np.uint8)
            im = im.reshape((image_msg.height, image_msg.width, 3))
            # plt.imshow(im)
            # plt.savefig(f"frames/frame{sim_step:0>4d}.jpg")
            img_time = image_msg.header.stamp.sec + image_msg.header.stamp.nsec*1e-9

            # ensure image message and simulator are in sync
            assert sim_time == img_time

gazebo.close()
