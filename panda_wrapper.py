from scenario import gazebo as scenario_gazebo
from scenario import core as scenario_core
import gym_ignition_environments
from gym_ignition.context.gazebo import controllers
from panda_controller import LinearJointSpacePlanner


import numpy as np
import numpy.typing as npt


class Panda(LinearJointSpacePlanner, gym_ignition_environments.models.panda.Panda):
    def __init__(self, **kwargs):
        self.home_position = np.array((0, -0.785,0, -2.356, 0, 1.571, 0.785, 0.03, 0.03))
        super().__init__(**kwargs)

        # Constraints

        # joint constraints (units in rad, e.g. rad/s for velocity)
        # TODO: check the values of the fingers, these are all guesses
        self.max_position = np.array((2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973, 0.041, 0.041))
        self.min_position = np.array((-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.001, -0.001))
        self.max_velocity = np.array((2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 0.3, 0.3))
        self.min_velocity = - self.max_velocity
        self.max_acceleration = np.array((15, 7.5, 10, 12.5, 15, 20, 20, 10, 10), dtype=np.float_)
        self.min_acceleration = - self.max_acceleration
        self.max_jerk = np.array((7500, 3750, 5000, 6250, 7500, 10000, 10000, 10000, 10000), dtype=np.float_)
        self.min_jerk = - self.max_jerk
        self.max_torque = np.array((87, 87, 87, 87, 12, 12, 12, 12, 12), dtype=np.float_)
        self.min_torque = - self.max_torque
        self.max_rotatum = np.array([1000] * 9)
        self.min_rotatum = - self.max_rotatum

        # tool constraints
        self.max_tool_velocity = 1.7  # m/s
        self.max_tool_acceleration = 13  # m/s
        self.max_tool_jerk = 6500  # m/s
        self.max_tool_angular_velocity = 2.5  # rad/s
        self.max_tool_angular_acceleration = 25  # rad/s
        self.max_tool_angular_jerk = 12500  # rad/s

        # ellbow constraints (in rad)
        # This is in the docs, but I'm not sure how to interpret it. Perhaps it
        # refers to null-space motion?
        # https://frankaemika.github.io/docs/control_parameters.html
        self.max_ellbow_velocity = 2.175
        self.max_ellbow_acceleration = 10
        self.max_ellbow_jerk = 5000

        panda = self.model

        panda.to_gazebo().enable_self_collisions(True)

        # Insert the ComputedTorqueFixedBase controller
        assert panda.to_gazebo().insert_model_plugin(
            *controllers.ComputedTorqueFixedBase(
                kp=[100.0] * (self.dofs - 2) + [10000.0] * 2,
                ki=[0.0] * self.dofs,
                kd=[17.5] * (self.dofs - 2) + [100.0] * 2,
                urdf=self.get_model_file(),
                joints=self.joint_names(),
            ).args()
        )

    def reset(self):
        self.position = self.home_position
        self.velocity = [0] * 9
        self.target_position = self.home_position
        self.target_velocity = [0] * 9
        self.target_acceleration = [0] * 9

    

    @property
    def dofs(self):
        return self.model.dofs()

    @property
    def position(self):
        return np.array(self.model.joint_positions())

    @property
    def velocity(self):
        return np.array(self.model.joint_velocities())

    @property
    def acceleration(self):
        return np.array(self.model.joint_accelerations())

    @position.setter
    def position(self, position: npt.ArrayLike):
        position = np.asarray(position)

        if np.any((position < self.min_position) | (self.max_position < position)):
            raise ValueError("The position exceeds the robot's limits.")

        assert self.model.to_gazebo().reset_joint_positions(position.tolist())

    @velocity.setter
    def velocity(self, velocity: npt.ArrayLike):
        velocity = np.asarray(velocity)

        if np.any((velocity < self.min_velocity) | (self.max_velocity < velocity)):
            raise ValueError("The velocity exceeds the robot's limits.")

        assert self.model.to_gazebo().reset_joint_velocities(velocity.tolist())

    @property
    def target_position(self):
        return np.array(self.model.joint_position_targets())

    @property
    def target_velocity(self):
        return np.array(self.model.joint_velocity_targets())

    @property
    def target_acceleration(self):
        return np.array(self.model.joint_acceleration_targets())

    @target_position.setter
    def target_position(self, position: npt.ArrayLike):
        position = np.asarray(position)

        if np.any((position < self.min_position) | (self.max_position < position)):
            raise ValueError("The target position exceeds the robot's limits.")

        assert self.model.set_joint_position_targets(position.tolist())

    @target_velocity.setter
    def target_velocity(self, velocity: npt.ArrayLike):
        velocity = np.asarray(velocity)

        if np.any((velocity < self.min_velocity) | (self.max_velocity < velocity)):
            raise ValueError("The target velocity exceeds the robot's limits.")

        assert self.model.set_joint_velocity_targets(velocity.tolist())

    @target_acceleration.setter
    def target_acceleration(self, acceleration: npt.ArrayLike):
        acceleration = np.asarray(acceleration)

        if np.any((acceleration < self.min_acceleration) | (self.max_acceleration < acceleration)):
            raise ValueError("The target acceleration exceeds the robot's limits.")

        assert self.model.set_joint_acceleration_targets(acceleration.tolist())

    @property
    def tool(self):
        return self.model.get_link("end_effector_frame")

    @property
    def tool_pose(self):
        position = self.model.get_link("end_effector_frame").position()
        orientation = self.model.get_link("end_effector_frame").orientation()

        return (position, orientation)

    @tool_pose.setter
    def tool_pose(self, pose):
        """Set the joints so that the tool is in the desired configuration"""

        position, orientation = pose
        if position is None and orientation is None:
            return

        self.position = self.solve_ik(position=position, orientation=orientation)

    # @target_tool_pose.setter
    def target_tool_pose(self, pose):
        position, orientation = pose
        self.target_position = self.solve_ik(position=position, orientation=orientation)

    def solve_ik(self, *, position=None, orientation=None):
        if position is None and orientation is None:
            return self.position

        old_position, old_orientation = self.tool_pose
        position = old_position if position is None else position
        orientation = old_orientation if orientation is None else orientation

        # reset IK
        self.ik.set_current_robot_configuration(
            base_position=np.array(self.base_position()),
            base_quaternion=np.array(self.base_orientation()),
            joint_configuration=self.home_position,
        )
        self.ik.solve()

        return super().solve_ik(position, orientation)

    @property
    def tool_velocity(self):
        return self.model.get_link("end_effector_frame").world_linear_velocity()

    @property
    def tool_angular_velocity(self):
        return self.model.get_link("end_effector_frame").world_angular_velocity()

    @property
    def tool_acceleration(self):
        return self.model.get_link("end_effector_frame").world_linear_acceleration()

    @property
    def tool_angular_acceleration(self):
        return self.model.get_link("end_effector_frame").world_angular_acceleration()

class PandaMixin:
    """Add a Panda Robot to the simulator"""

    def __init__(self, *, panda_config, **kwargs):
        super().__init__(**kwargs)

        self.panda = None
        self.config = panda_config

    def initialize(self):
        super().initialize()
        self.config["world"] = self.world
        self.panda = Panda(**self.config)
        panda = self.panda
        assert panda.set_controller_period(period=self.step_size())

        panda.reset()
        self.run(paused=True)  # update the controller positions
        panda.target_position = panda.position

        home_position = np.array(panda.tool.position())
        home_orientation = np.array(panda.tool.orientation())
        home_pose = np.hstack((home_position, home_orientation[[1, 2, 3, 0]]))
