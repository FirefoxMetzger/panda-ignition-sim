from scenario import gazebo as scenario_gazebo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
import ropy.transform as tf
import ropy.ignition as ign


def camera_parser(msg):
    image_msg = ign.messages.Image()
    image_msg.parse(msg[2])

    image = np.frombuffer(image_msg.data, dtype=np.uint8)
    image = image.reshape((image_msg.height, image_msg.width, 3))

    return image


gazebo = scenario_gazebo.GazeboSimulator(step_size=0.001, rtf=1.0, steps_per_run=1)
assert gazebo.insert_world_from_sdf("./world1.sdf")
gazebo.initialize()

# Fix: available topics seem to only be updated at the end
# of a step. This allows the subscribers to find the topic's
# address
gazebo.run(paused=True)

# known a-priori from .sdf
# would be nice if we could read this from gazebo directly
width = 320
height = 240
hFoV = 1.047

# get extrinsic matrix
camera = gazebo.get_world("camera_sensor").get_model("camera").get_link("link")
cam_pos_world = np.array(camera.position())
cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
extrinsic_transform = tf.coordinates.transform(camera_frame_world)

# get intrinsic matrix
intrinsic_transform = tf.projections.camera_frustum(hFoV, (height, width))

# scene objects
box = gazebo.get_world("camera_sensor").get_model("box")

with ign.Subscriber("/camera", parser=camera_parser) as camera_topic:
    # Fix: the first step doesn't generate messages.
    # I don't exactly know why; I assume it has
    # to do with subscriptions being updated at the end
    # of the sim loop instead of the beginning?
    gazebo.run(paused=True)

    img = camera_topic.recv()

    # project cone
    box_corner = np.array(box.base_position()) + np.array((0.5, -0.5, 0.5))
    pos_world = tf.homogenize(box_corner)
    pos_cam = np.matmul(extrinsic_transform, pos_world)
    pos_px_hom = np.matmul(intrinsic_transform, pos_cam)
    cube_pos_px = tf.cartesianize(pos_px_hom)

    # visualize
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.add_patch(Circle(cube_pos_px, radius=6))
    plt.show()

gazebo.close()
