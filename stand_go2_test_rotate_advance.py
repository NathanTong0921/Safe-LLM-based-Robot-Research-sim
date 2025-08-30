import time
import sys
import numpy as np
from math import sin, cos, pi, atan2, sqrt, acos

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

stand_up_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763
], dtype=float)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375,
   -0.0473455, 1.22187, -2.44375,
    0.0473455, 1.22187, -2.44375,
   -0.0473455, 1.22187, -2.44375
], dtype=float)

# 关节编号：hip, thigh, calf
# 每条腿：LF(0~2), RF(3~5), LB(6~8), RB(9~11)
LF = [0, 1, 2]
RF = [3, 4, 5]
LB = [6, 7, 8]
RB = [9, 10, 11]

# === IK 模型参数（根据 Mini Cheetah 实际参数或近似）===
L1 = 0.213 # thigh link [m]
L2 = 0.213 # calf link [m]

dt = 0.002
runing_time = 0.0
crc = CRC()
omega = 2 * np.pi * 2
yaw_rate_x = -0.3
yaw_rate_z = 0.1
hip_amp = 0.00          # 可选：动态摆动
hip_yaw_gain = 0.0     # yaw 旋转控制 hip 偏移的增益因子

# 🦿 每条腿相对机身中心的位置，用于 yaw 转圈时的横向偏移计算
leg_offsets = {
    "LF": np.array([ 0.2,  0.1]),
    "RF": np.array([ 0.2, -0.1]),
    "LB": np.array([-0.2,  0.1]),
    "RB": np.array([-0.2, -0.1]),
}

def generate_trot_trajectory(t, phase_offset=0.0, step_height=0.06, step_length=0.05):
    # omega = 2 * np.pi * 1.5  # 频率1.5Hz
    phase = omega * t + phase_offset
    z = -step_height * max(0, sin(phase)) + 0.35  # 抬腿（只有正半周期）
    x = -step_length * sin(phase)      # 前后摆
    return x, z

def leg_inverse_kinematics(x, z):
    # # 简化 2D IK 解法：给出足端位置 [x, z]，输出 joint angles
    # dist = sqrt(x**2 + z**2)
    # if dist > (L1 + L2 - 1e-3):  # reach limit
    #     dist = L1 + L2 - 1e-3
    # angle_knee = pi - acos((L1**2 + L2**2 - dist**2) / (2 * L1 * L2))
    # angle_thigh = -atan2(x, -z) - acos((L1**2 + dist**2 - L2**2) / (2 * L1 * dist))
    #     # 限制角度范围，避免过大摆动
    # # angle_thigh = max(min(angle_thigh, 0.2), -0.2)  # +-0.2 rad 约11度
    # # angle_knee = max(min(angle_knee, 0.5), 1.5)     # 0.5~1.5 rad 合理范围
    # return angle_thigh, -angle_knee
        # x: 前后方向，z: 垂直方向（注意 z 要为负）
    l = sqrt(x**2 + z**2)
    l = min(l, L1 + L2 - 1e-4)

    # 夹角公式
    cos_knee = (L1**2 + L2**2 - l**2) / (2 * L1 * L2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    angle_knee = pi - acos(cos_knee)

    # 计算大腿角
    cos_thigh = (L1**2 + l**2 - L2**2) / (2 * L1 * l)
    cos_thigh = np.clip(cos_thigh, -1.0, 1.0)
    angle_thigh_offset = acos(cos_thigh)
    angle_thigh = (np.pi/2)- (atan2(z, x) - angle_thigh_offset)

    return angle_thigh, -angle_knee

# 🐾 步态相位（对角线同步）
gait_phase_offsets = {
    "LF": 0.0,
    "RF": pi,
    "LB": pi,
    "RB": 0.0
}

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0

    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    while True:
        step_start = time.perf_counter()
        runing_time += dt

        if runing_time < 3.0:
            # 0–3s: 站立
            phase = np.tanh(runing_time / 1.2)
            for i in range(12):
                cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (1 - phase) * stand_down_joint_pos[i]
                cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 3.5
                cmd.motor_cmd[i].tau = 0.0

        elif runing_time < 10.0:

            # 🐾 Trot gait period: generate foot trajectory + IK
            foot_targets = {}
            for leg_name in ["LF", "RF", "LB", "RB"]:
                x, z = generate_trot_trajectory(runing_time - 3.0, phase_offset=gait_phase_offsets[leg_name])
                if leg_name in ['LB', 'RB']:
                    z += 0.02 # 后腿抬得更低，增强支撑力，减少身体后仰
                
                leg_offset = leg_offsets[leg_name]
                dx = yaw_rate_x * leg_offset[1]
                dy =  - yaw_rate_z * leg_offset[0]

                thigh, calf = leg_inverse_kinematics(x + dx, z)
                foot_targets[leg_name] = (dy, thigh, calf)

                # 打印输出
                print(f"[{leg_name}] x = {x:.4f}, z = {z:.4f}, thigh = {thigh:.4f}, calf = {calf:.4f}")
            


            joint_map = {
                "LF": [0, 1, 2],
                "RF": [3, 4, 5],
                "LB": [6, 7, 8],
                "RB": [9, 10, 11]
            }

            for leg_name, (dy, thigh, calf) in foot_targets.items():
                idx = joint_map[leg_name]
                # 保持 hip 为站立角度，添加小量摆动也可以
                # stand_up_joint_pos_mod = stand_up_joint_pos.copy()
                # stand_up_joint_pos_mod[LB[0]] -= 0.02  # LB hip 向前摆
                # stand_up_joint_pos_mod[RB[0]] -= 0.02  # RB hip 向前摆
                
                # yaw 引起的 hip 偏移
                hip_yaw_offset = dy * hip_yaw_gain
                #  hip 动态摆动（可为 0）
                hip_dynamic  = hip_amp * sin(omega * (runing_time - 3.0) + gait_phase_offsets[leg_name])

                cmd.motor_cmd[idx[0]].q = stand_up_joint_pos[idx[0]]+ hip_dynamic + hip_yaw_offset 
                cmd.motor_cmd[idx[1]].q = thigh
                cmd.motor_cmd[idx[2]].q = calf
                for j in idx:
                    cmd.motor_cmd[j].kp = 50.0
                    cmd.motor_cmd[j].kd = 2
                    cmd.motor_cmd[j].dq = 0.0
                    cmd.motor_cmd[j].tau = 0.0

        else:
            # 6s+: 保持站立
            # for i in range(12):
            #     cmd.motor_cmd[i].q = stand_up_joint_pos[i]
            #     cmd.motor_cmd[i].kp = 50.0
            #     cmd.motor_cmd[i].dq = 0.0
            #     cmd.motor_cmd[i].kd = 3.5
            #     cmd.motor_cmd[i].tau = 0.0

                            # Then stand down
            phase = np.tanh((runing_time - 10.0) / 1.2)
            for i in range(12):
                cmd.motor_cmd[i].q = phase * stand_down_joint_pos[i] + (
                    1 - phase) * stand_up_joint_pos[i]
                cmd.motor_cmd[i].kp = 50.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 1
                cmd.motor_cmd[i].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
