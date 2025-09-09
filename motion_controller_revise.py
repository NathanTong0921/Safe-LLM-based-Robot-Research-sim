import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import numpy as np
import sys
from math import sin, cos, pi, atan2, sqrt, acos

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_


switch_state =False
stand_up_joint_pos = np.array([0.0, 0.67, -1.3] * 4, dtype=np.float32)
stand_down_joint_pos = np.array([0.0, 1.36, -2.65] * 4, dtype=np.float32)
NUM_MOTORS = 12

STANDING_KP = 1000.0
STANDING_KD = 10.0
WALKING_KP = 600.0
WALKING_KD = 20.0
TURNING_KP=950.0
TURNING_KD=20.0

L1 = 0.35  
L2 = 0.35  
H=0.12
STEP_HEIGHT = 0.15
STEP_LENGTH = 0.10
TURNING_STEP_HEIGHT = 0.08  
TURNING_STEP_LENGTH = 0.03   
OMEGA = 2 * np.pi * 1.2
# HIP_OPEN_ANGLE = 0.1
# HIP_SWAY = 0.1

LEG_JOINT_MAP = {
    "FR": (0, 1, 2), 
    "FL": (3, 4, 5), 
    "RR": (6, 7, 8), 
    "RL": (9, 10, 11)
}
GAIT_PHASE_OFFSETS = {
    "FR": 0.0, 
    "FL": pi, 
    "RR": pi, 
    "RL": 0.0
}

YAW_RATE = 0.4
LEG_OFFSETS = {
    "FR": np.array([0.2, -0.1]), 
    "FL": np.array([0.2, 0.1]),
    "RR": np.array([-0.2, -0.1]),
    "RL": np.array([-0.2, 0.1]),
}

DX_GAIN = 0.15                 
TURN_BLEND_RISE_TIME = 0.30    
TURN_BLEND_FALL_TIME = 0.30    
YAW_RATE_RAMP = 2.0            
Q_FILTER_TAU = 0.05            
MAX_Q_STEP = 0.20             

def generate_trot_trajectory(t, phase_offset, omega_val, step_height_val, step_length_val):
    phase = omega_val * t + phase_offset
    z = -step_height_val * max(0, sin(phase)) + 0.55
    x = -step_length_val * sin(phase)
    return x, z

def leg_inverse_kinematics(x, z, l1_val, l2_val):
    l = sqrt(x**2 + z**2)
    l = min(l, l1_val + l2_val - 1e-4)
    cos_knee = (l1_val**2 + l2_val**2 - l**2) / (2 * l1_val * l2_val)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    angle_knee = -(pi - acos(cos_knee)) 
    cos_thigh_part = (l1_val**2 + l**2 - l2_val**2) / (2 * l1_val * l)
    cos_thigh_part = np.clip(cos_thigh_part, -1.0, 1.0)
    angle_thigh_offset = acos(cos_thigh_part)
    angle_thigh_base = (np.pi/2)- (atan2(z, x) - angle_thigh_offset)
    return angle_thigh_base, angle_knee

class MotionControllerNode(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.get_logger().info('B2 Motion Controller Initializing...')
        ChannelFactoryInitialize(1, "lo")
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()
        self.command_subscriber = self.create_subscription(String, 'robot_command', self.command_callback, 10)
        self.dt = 0.002
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.robot_state = "SITTING"
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.transition_start_time = 0.0
        self.walk_start_time = 0.0
        self.yaw_rate_target = 0.0
        self.yaw_rate_current = 0.0
        self.turn_blend = 0.0
        self.last_time = time.time()
        self.prev_q_des = stand_up_joint_pos.copy()
        self.get_logger().info("Node is ready. Send 'stand', 'walk', 'turn_left', 'turn_right', 'stop', 'sit' commands.")

    def command_callback(self, msg: String):
        self.get_logger().info(f'Received ROS 2 command: "{msg.data}"')
        current_state = self.robot_state
        command = msg.data
        if command == "stand" and current_state == "SITTING":
            self.robot_state = "STANDING_UP"
            self.transition_start_time = time.time()
        elif command == "walk" and current_state in ["STANDING", "WALKING", "TURNING_LEFT", "TURNING_RIGHT"]:
            if current_state == "STANDING" or self.walk_start_time == 0.0:
                self.walk_start_time = time.time()  
            self.yaw_rate_target = 0.0
            self.robot_state = "WALKING"
            self.get_logger().info('State changed to: WALKING')
        elif command == "turn_left" and current_state in ["STANDING", "WALKING", "TURNING_RIGHT"]:
            if current_state == "STANDING" or self.walk_start_time == 0.0:
                self.walk_start_time = time.time()
            self.yaw_rate_target = -YAW_RATE
            self.robot_state = "TURNING_LEFT"
            self.get_logger().info('State changed to: TURNING_LEFT')
        elif command == "turn_right" and current_state in ["STANDING", "WALKING", "TURNING_LEFT"]:
            if current_state == "STANDING" or self.walk_start_time == 0.0:
                self.walk_start_time = time.time()
            self.yaw_rate_target = YAW_RATE
            self.robot_state = "TURNING_RIGHT"
            self.get_logger().info('State changed to: TURNING_RIGHT')
        elif command == "stop" and current_state in ["WALKING", "TURNING_LEFT", "TURNING_RIGHT"]:
            self.yaw_rate_target = 0.0
            self.robot_state = "STANDING"
            self.get_logger().info('State changed to: STANDING')
        elif command == "sit" and current_state == "STANDING":
            self.robot_state = "SITTING_DOWN"
            self.transition_start_time = time.time()
    
    def _update_smooth_vars(self, dt: float):
        diff = self.yaw_rate_target - self.yaw_rate_current
        max_step = YAW_RATE_RAMP * max(dt, 1e-6)
        if diff > max_step:
            diff = max_step
        elif diff < -max_step:
            diff = -max_step
        self.yaw_rate_current += diff

        target_blend = 1.0 if self.robot_state in ["TURNING_LEFT", "TURNING_RIGHT"] else 0.0
        if target_blend > self.turn_blend:
            rate = dt / max(TURN_BLEND_RISE_TIME, 1e-6)
        else:
            rate = dt / max(TURN_BLEND_FALL_TIME, 1e-6)
        self.turn_blend += (target_blend - self.turn_blend) * min(rate, 1.0)
        self.turn_blend = max(0.0, min(1.0, self.turn_blend))

    def _filter_joint_targets(self, q_raw: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0.0:
            return q_raw
        alpha = dt / max(Q_FILTER_TAU, dt)
        q_lp = self.prev_q_des + alpha * (q_raw - self.prev_q_des)
        delta = np.clip(q_lp - self.prev_q_des, -MAX_Q_STEP, MAX_Q_STEP)
        q_out = self.prev_q_des + delta
        self.prev_q_des = q_out.copy()
        return q_out

    def control_loop(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self._update_smooth_vars(dt)

        if self.robot_state == "STANDING_UP":
            duration = 3.0
            phase = min((current_time - self.transition_start_time) / duration, 1.0)
            target_pos = phase * stand_up_joint_pos + (1 - phase) * stand_down_joint_pos
            if phase >= 1.0:
                self.robot_state = "STANDING"
                self.get_logger().info('State changed to: STANDING')

            for i in range(NUM_MOTORS):
                self.cmd.motor_cmd[i].q = float(target_pos[i])
                self.cmd.motor_cmd[i].kp = STANDING_KP
                self.cmd.motor_cmd[i].kd = STANDING_KD
                self.cmd.motor_cmd[i].mode = 1

        elif self.robot_state == "STANDING" or self.robot_state == "SITTING":
            if self.robot_state == "STANDING":
                target_pos = stand_up_joint_pos
            else:
                target_pos = stand_down_joint_pos

            for i in range(NUM_MOTORS):
                self.cmd.motor_cmd[i].q = float(target_pos[i])
                self.cmd.motor_cmd[i].kp = STANDING_KP
                self.cmd.motor_cmd[i].kd = STANDING_KD
                self.cmd.motor_cmd[i].mode = 1


        elif self.robot_state == "SITTING_DOWN":
            duration = 3.0
            phase = min((current_time - self.transition_start_time) / duration, 1.0)
            target_pos = phase * stand_down_joint_pos + (1 - phase) * stand_up_joint_pos
            if phase >= 1.0:
                self.robot_state = "SITTING"
                self.get_logger().info('State changed to: SITTING')

            for i in range(NUM_MOTORS):
                self.cmd.motor_cmd[i].q = float(target_pos[i])
                self.cmd.motor_cmd[i].kp = STANDING_KP
                self.cmd.motor_cmd[i].kd = STANDING_KD
                self.cmd.motor_cmd[i].mode = 1

        elif self.robot_state == "WALKING":
            walk_time = current_time - self.walk_start_time
            q_target = np.zeros(NUM_MOTORS, dtype=np.float32)
            for leg_name, joint_indices in LEG_JOINT_MAP.items():
                hip_idx, thigh_idx, calf_idx = joint_indices
                x, z = generate_trot_trajectory(
                    walk_time, GAIT_PHASE_OFFSETS[leg_name],
                    omega_val=OMEGA,
                    step_height_val=STEP_HEIGHT,
                    step_length_val=STEP_LENGTH)
                thigh_q, calf_q = leg_inverse_kinematics(x, z, l1_val=L1, l2_val=L2)
                q_target[hip_idx] = stand_up_joint_pos[hip_idx]
                q_target[thigh_idx] = float(thigh_q)
                q_target[calf_idx] = float(calf_q)

            q_cmd = self._filter_joint_targets(q_target, dt)
            for i in range(NUM_MOTORS):
                self.cmd.motor_cmd[i].q = float(q_cmd[i])
                self.cmd.motor_cmd[i].kp = WALKING_KP
                self.cmd.motor_cmd[i].kd = WALKING_KD
                self.cmd.motor_cmd[i].mode = 1

        elif self.robot_state in ["TURNING_LEFT", "TURNING_RIGHT"]:
            walk_time = current_time - self.walk_start_time
            turning_step_len = max(0.6 * STEP_LENGTH, 0.03)
            q_target = np.zeros(NUM_MOTORS, dtype=np.float32)

            for leg_name, joint_indices in LEG_JOINT_MAP.items():
                hip_idx, thigh_idx, calf_idx = joint_indices
                x, z = generate_trot_trajectory(
                    walk_time, GAIT_PHASE_OFFSETS[leg_name],
                    omega_val=OMEGA,
                    step_height_val=TURNING_STEP_HEIGHT,
                    step_length_val=turning_step_len)

                leg_offset = LEG_OFFSETS[leg_name]
                dx = self.yaw_rate_current * leg_offset[1] * DX_GAIN
                x += self.turn_blend * dx

                thigh_q, calf_q = leg_inverse_kinematics(x, z, l1_val=L1, l2_val=L2)
                q_target[hip_idx] = stand_up_joint_pos[hip_idx]
                q_target[thigh_idx] = float(thigh_q)
                q_target[calf_idx] = float(calf_q)

            q_cmd = self._filter_joint_targets(q_target, dt)
            for i in range(NUM_MOTORS):
                self.cmd.motor_cmd[i].q = float(q_cmd[i])
                self.cmd.motor_cmd[i].kp = TURNING_KP
                self.cmd.motor_cmd[i].kd = TURNING_KD
                self.cmd.motor_cmd[i].mode = 1

        self.low_cmd_publisher.Write(self.cmd)
        

def main(args=None):
    rclpy.init(args=args)
    controller_node = MotionControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()