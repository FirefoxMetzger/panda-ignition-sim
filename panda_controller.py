from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from scenario import core as scenario_core
from gym_ignition.rbd import conversions
from itertools import chain


from scipy.spatial.transform import Rotation as R
from ropy.trajectory import spline_trajectory, linear_trajectory
import numpy as np


class LinearJointSpacePlanner:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ik_joints = [
            j.name()
            for j in self.model.joints()
            if j.type is not scenario_core.JointType_fixed and "finger" not in j.name()
        ]

        # Initialize Panda in Ignition
        assert (
            self.get_joint(joint_name="panda_finger_joint1")
            .to_gazebo()
            .set_max_generalized_force(max_force=500.0)
        )
        assert (
            self.get_joint(joint_name="panda_finger_joint2")
            .to_gazebo()
            .set_max_generalized_force(max_force=500.0)
        )

        # Initialize IK

        ik_joints = self.ik_joints
        ik = inverse_kinematics_nlp.InverseKinematicsNLP(
            urdf_filename=self.get_model_file(),
            considered_joints=ik_joints,
            joint_serialization=self.joint_names(),
        )

        ik.initialize(
            verbosity=0,
            floating_base=False,
            cost_tolerance=1e-8,
            constraints_tolerance=1e-8,
            base_frame=self.base_frame(),
        )

        ik.set_current_robot_configuration(
            base_position=np.array(self.base_position()),
            base_quaternion=np.array(self.base_orientation()),
            joint_configuration=self.home_position,
        )

        ik.add_target(
            frame_name="end_effector_frame",
            target_type=inverse_kinematics_nlp.TargetType.POSE,
            as_constraint=False,
        )
        ik.solve()  # warm up IK
        self.ik = ik

    def solve_ik(
        self, target_position: np.ndarray, target_orientation: np.array
    ) -> np.ndarray:
        self.ik.update_transform_target(
            target_name=self.ik.get_active_target_names()[0],
            position=target_position,
            quaternion=np.array(target_orientation),
        )

        # Run the IK
        self.ik.solve()
        result = self.position.copy()
        result[:-2] = self.ik.get_reduced_solution().joint_configuration

        return result

    def plan(self, t, pose_key, *, t_key=None, t_begin=0, t_end=1):
        if t_key is None:
            t_key = [
                np.linspace(t_begin, t_end, len(p), dtype=np.float_) for p in pose_key
            ]

        all_times = np.array([t for t in chain.from_iterable(t_key)])
        all_times = np.unique(all_times)
        all_times.sort()

        positions = spline_trajectory(
            all_times, pose_key[0], t_control=t_key[0], t_min=t_begin, t_max=t_end
        )
        orientations = linear_trajectory(
            all_times, pose_key[1], t_control=t_key[1], t_min=t_begin, t_max=t_end
        )

        keyframes_jointspace = np.empty(shape=(len(all_times), 9))
        for idx, (pos, ori) in enumerate(zip(positions, orientations)):
            keyframes_jointspace[idx] = self.solve_ik(pos, ori)

        pose = spline_trajectory(
            t, keyframes_jointspace, t_control=all_times, t_min=t_begin, t_max=t_end
        )
        twist = spline_trajectory(
            t,
            keyframes_jointspace,
            t_control=all_times,
            t_min=t_begin,
            t_max=t_end,
            derivative=1,
        )
        wrench = spline_trajectory(
            t,
            keyframes_jointspace,
            t_control=all_times,
            t_min=t_begin,
            t_max=t_end,
            derivative=2,
        )

        return np.stack((pose, twist, wrench), axis=-1)


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
            The number of derivatives to compute for the motion. The default is
            3. The derivatives are stacked along the last axis of the returned
            array. When planing for positions only, ths default will compute
            (position, velocity, acceleration, jerk) and produce a result where,
            e.g., ``acceleration = plan(t, ...)[..., 2]``.
        **kwargs : any
            Other, non-standard keyword arguments
        """
        raise NotImplementedError
