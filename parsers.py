"""
A collection of parsers used to deserialize messages used in the simulator
"""

from ignition.msgs.image_pb2 import Image
from ignition.msgs.clock_pb2 import Clock
from dataclasses import dataclass
import numpy as np

@dataclass
class ImageMessage:
    image : np.array
    time : float

def clock_parser(msg):
    clock_msg = Clock()
    clock_msg.ParseFromString(msg[2])
    sim_time = clock_msg.sim.sec + clock_msg.sim.nsec*1e-9
    return sim_time

def camera_parser(msg):
    image_msg = Image()
    image_msg.ParseFromString(msg[2])

    im = np.frombuffer(image_msg.data, dtype=np.uint8)
    im = im.reshape((image_msg.height, image_msg.width, 3))

    img_time = image_msg.header.stamp.sec + image_msg.header.stamp.nsec*1e-9

    return ImageMessage(image=im,time=img_time)
