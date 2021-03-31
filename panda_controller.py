from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scenario import core as scenario_core
from gym_ignition.rbd import conversions


from scipy.spatial.transform import Rotation as R
from ropy.trajectory import linear_trajectory
import numpy as np


class LinearJointSpacePlanner:
    def __init__(self, panda, control_frequency=0.001):
        # Initialize Panda in Ignition
        self.panda = panda
        assert panda.set_controller_period(period=control_frequency)
        assert (
            panda.get_joint(joint_name="panda_finger_joint1")
            .to_gazebo()
            .set_max_generalized_force(max_force=500.0)
        )
        assert (
            panda.get_joint(joint_name="panda_finger_joint2")
            .to_gazebo()
            .set_max_generalized_force(max_force=500.0)
        )

        # Initialize IK

        ik_joints = [
            j.name()
            for j in panda.joints()
            if j.type is not scenario_core.JointType_fixed
        ]
        ik = inverse_kinematics_nlp.InverseKinematicsNLP(
            urdf_filename=panda.get_model_file(),
            considered_joints=ik_joints,
            joint_serialization=panda.joint_names(),
        )

        ik.initialize(
            verbosity=1,
            floating_base=False,
            cost_tolerance=1e-8,
            constraints_tolerance=1e-8,
            base_frame=panda.base_frame(),
        )

        ik.set_current_robot_configuration(
            base_position=np.array(panda.base_position()),
            base_quaternion=np.array(panda.base_orientation()),
            joint_configuration=np.array(panda.joint_positions()),
        )

        ik.add_target(
            frame_name="end_effector_frame",
            target_type=inverse_kinematics_nlp.TargetType.POSE,
            as_constraint=False,
        )
        self.ik = ik

    def solve_ik(self, target_position: np.ndarray, target_orientation: np.array) -> np.ndarray:
        self.ik.update_transform_target(
            target_name=self.ik.get_active_target_names()[0],
            position=target_position,
            quaternion=target_orientation,
        )

        # Run the IK
        self.ik.solve()

        return self.ik.get_reduced_solution().joint_configuration

    def plan(self, t, keyframes, *, t_key=None, t_begin=0, t_end=1):
        keyframes = np.asarray(keyframes)
        positions = keyframes[:, :3]
        orientations = keyframes[:, 3:]
        orientations = orientations[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        keyframes_jointspace = np.empty(shape=(keyframes.shape[0], 9))
        for idx, (pos, ori) in enumerate(zip(positions, orientations)):
            keyframes_jointspace[idx] = self.solve_ik(pos, ori)

        return linear_trajectory(
            t, keyframes_jointspace, t_control=t_key, t_min=t_begin, t_max=t_end
        )


class MotionPlanner:
    def plan(
        self, t, keyframes, *, t_key=None, t_begin=0, t_end=1, derivatives=3, **kwargs
    ):
        """Plans a motion through keyframes represented by points at times t.

        Parameters
        ----------
        t : np.array
            The time points at which to evaluate the resulting plan. For
            controllers operating at a fixed frequency, this is
            ``np.arange(t_min, t_max+1/frequency, step=1/frequency)``
        keyframes : np.array
            A batch of keyframes of the motion. The first dimension is the batch
            dimension. The batch should contain at least two keyframes.Keyframes
            are sometimes also referred to as via-points or control points.
        t_key : np.array
            The time point at which each keyframe should be reached. If None,
            they will be spaced out evenly within ``[t_min, t_max]``. If not None,
            the following inequality must hold ``t_key[0]<=t_min<t_max<=t_key[-1]``.
        t_begin : float
            The starting time of the motion. The starting point is
            ``pos=plan(t_begin)``.
        t_end : float
            The end time of the motion. The end point will be
            ``pos=plan(t_end)``.
        derivatives : int, optional
            The number of derivatives to compute for the motion. The default is 3,
            which will compute (position, velocity, acceleration, jerk). The derivatives
            are stacked along the last axis of the returned array, e.g.,
            ``acceleration = plan(t, ...)[..., 2]``.
        **kwargs : any
            Other, non-standard keyword arguments
        """
        raise NotImplementedError
