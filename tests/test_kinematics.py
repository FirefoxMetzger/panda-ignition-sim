import numpy as np
import kinematics as kine
import pytest


@pytest.mark.parametrize(
    "vector_in,frame_A,frame_B,vector_out",
    [
        (np.array((1,2,3)),np.array((0,0,0,0,0,0)),np.array((1,0,0,0,0,0)),np.array((0,2,3))),
        (np.array((1,0,0)),np.array((0,0,0,0,0,0)),np.array((0,0,0,0,0,np.pi/2)),np.array((0,1,0))),
        (np.array((0,1,0)),np.array((0,0,0,0,0,0)),np.array((0,0,0,np.pi/2,0,np.pi/2)),np.array((0,0,1))),
        (np.array((0,np.sqrt(2),0)),np.array((4.5,1,0,0,0,-np.pi/4)),np.array((0,0,0,0,0,0)),np.array((3.5,2,0))),
    ]
)
def test_transform_between(vector_in, frame_A, frame_B, vector_out):
    vector_A = kine.homogenize(vector_in)
    vector_B = np.matmul(kine.transformBetween(frame_A, frame_B), vector_A)
    vector_B = kine.cartesianize(vector_B)

    assert np.allclose(vector_B, vector_out)
