from scenario import gazebo as scenario_gazebo
import numpy as np
import ropy.ignition as ign
import matplotlib.pyplot as plt

gazebo = scenario_gazebo.GazeboSimulator(step_size=0.001, rtf=1.0, steps_per_run=1)
assert gazebo.insert_world_from_sdf("/usr/share/ignition/ignition-gazebo4/worlds/gpu_lidar_sensor.sdf")
gazebo.initialize()

# Fix: available topics seem to only be updated at the end
# of a step. This allows the subscribers to find the topic's
# address
gazebo.run(paused=True)

with ign.Subscriber("/lidar") as lidar:
    gazebo.run()
    lidar_msg = lidar.recv()

    # neat transition to numpy
    lidar_data = np.array(lidar_msg.ranges).reshape(
        (lidar_msg.vertical_count, lidar_msg.count)
    )
    lidar_data[lidar_data == np.inf] = lidar_msg.range_max

# has all the bells and whistles of a lidar message
print(
    f"""
Message type: {type(lidar_msg)}
Some more examples:
\tRange: ({lidar_msg.range_min},{lidar_msg.range_max})
\tReference Frame: {lidar_msg.frame}
\tData shape: {lidar_data.shape} 
"""
)

# visualize the lidar data 
# (not too meaningful I fear, but with some fantasy you can recognize the playground)
fig, ax = plt.subplots()
ax.imshow(lidar_data/lidar_data.max(), cmap="gray")
plt.show()
