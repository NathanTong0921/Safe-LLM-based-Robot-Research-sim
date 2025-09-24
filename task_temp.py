import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
import re
from enum import Enum, auto
import time
import math

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
# 添加 LowState 相关导入
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_

class NodeState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    DETECTING = auto()

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class TaskExecutorNode(Node):
    def __init__(self):
        super().__init__('task_executor_node')
        self.state = NodeState.IDLE
        self.actions = []

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_theta = 0.0  # 添加朝向角
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.tolerance = 0.3

        # Location for testing
        self.locations = {
            'entrance': (0.0, 0.0),
            'pepper_area': (3.5, 2.0),
            'fan_controller': (1.0, 4.0),
            'valve': (0.5, 1.5),
            'camera_tomato': (3.5, -2.0),
            'valve_south': (-1.0, -3.0),
            'sensor_cucumber': (2.0, 3.0)
        }
        self.get_logger().info('Location dictionary loaded.')
        
        ChannelFactoryInitialize(1, "lo")

        # 订阅高层状态（位置和速度）
        self.high_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.high_state_suber.Init(self.high_state_handler, 10)
        
        # 添加订阅低层状态（IMU 四元数）
        self.low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_suber.Init(self.low_state_handler, 10)

        self.ltl_plan_subscriber = self.create_subscription(
            String,
            'ltl_plan',
            self.ltl_plan_callback,
            10
        ) 
        self.cmd_publisher = self.create_publisher(String, 'robot_command', 10)
        # for real dog use Nav2
        #self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.detection_service_client = self.create_client(Trigger, 'trigger_detection')

        self.control_timer = None
        
        self.get_logger().info('TaskExecutor ready, state: IDLE')
        

    def ltl_plan_callback(self, msg):
        if self.state != NodeState.IDLE:
            self.get_logger().warn('Received LTL plan while not in IDLE state, ignoring.')
            return
            
        ltl_str = msg.data
        self.get_logger().info(f'Received LTL plan: {ltl_str}')
        self.actions = self.parse_ltl_plan(ltl_str)
        if self.actions:
            self.execute_next_action()
        else:
            self.get_logger().warn('No valid actions found in the LTL plan.')
    
    def execute_next_action(self):
        if not self.actions:
            self.get_logger().info('Task plan finished.')
            self.state = NodeState.IDLE
            self.get_logger().info('State changed: IDLE. Ready for new task.')
            return

        next_prop = self.actions.pop(0)
        self.get_logger().info(f'Executing proposition: {next_prop}')

        if next_prop.startswith('at_'):
            location_name = next_prop.replace('at_', '', 1)
            if location_name in self.locations: 
                self.state = NodeState.NAVIGATING
                self.get_logger().info(f'State changed: NAVIGATING to {location_name}')
                
                # replace here with new navigator
                x, y = self.locations[location_name]
                self.send_navigation_goal(float(x), float(y))
            else:
                self.get_logger().error(f'Unknown location: {location_name}. Skipping.')
                self.execute_next_action()

        elif next_prop.startswith('F(') or next_prop.startswith('G('):
            match = re.search(r'\((.*?)\)', next_prop)
            if not match:
                self.get_logger().error(f'Invalid action format: {next_prop}. Skipping.')
                self.execute_next_action()
                return
            
            action_name = match.group(1)
            self.get_logger().info(f'Interpreted action: {action_name}')
            
            # Here can call different services based on the action name
            # For now just handle a simulated "detection"
            if self.detection_service_client.service_is_ready():
                self.state = NodeState.DETECTING
                self.get_logger().info(f'State changed: DETECTING for action {action_name}')      
                request = Trigger.Request()
                future = self.detection_service_client.call_async(request)
                future.add_done_callback(self.detection_done_callback)
            else:
                self.get_logger().error('Detection service is not available! Skipping action.')
                self.execute_next_action()
        else:
            self.get_logger().warn(f'Unknown proposition format: {next_prop}. Skipping.')
            self.execute_next_action()

    def parse_ltl_plan(self, ltl_str: str) -> list:  
        try:      
            actions = [prop.strip() for prop in ltl_str.split('& X')]
            self.get_logger().info(f'Parsed propositions: {actions}')
            return actions
        except Exception as e:
            self.get_logger().error(f'Error when parsing LTL: {e}')
            return []

    # def send_navigation_goal(self, x, y):
    #     if not self.nav_action_client.wait_for_server(timeout_sec=3.0):
    #         self.get_logger().error('Nav2 server is not available')
    #         self.state = NodeState.IDLE
    #         return

    #     goal_msg = NavigateToPose.Goal()
    #     goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
    #     goal_msg.pose.header.frame_id = 'map' 
    #     goal_msg.pose.pose.position.x = x
    #     goal_msg.pose.pose.position.y = y
    #     goal_msg.pose.pose.orientation.w = 1.0 

    #     self.get_logger().info(f'Sending nav target to (x={x}, y={y})...')
    #     self._send_goal_future = self.nav_action_client.send_goal_async(
    #         goal_msg, feedback_callback=self.navigation_feedback_callback)
    #     self._send_goal_future.add_done_callback(self.goal_response_callback)

    # def goal_response_callback(self, future):
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().error('Rejected goal request by Nav2 server')
    #         self.state = NodeState.IDLE
    #         return

    #     self.get_logger().info('Goal accepted, executing...')
    #     self._get_result_future = goal_handle.get_result_async()
    #     self._get_result_future.add_done_callback(self.get_navigation_result_callback)

    # def navigation_feedback_callback(self, feedback_msg):
    #     feedback = feedback_msg.feedback
    #     self.get_logger().info(f'Navigation feedback: {feedback.distance_remaining:.2f} meters remaining.')

    # def get_navigation_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info(f'Navigation completed, result code: {result.code}')
    #     self.execute_next_action() #？

    def detection_done_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Received response from detection service: "{response.message}"')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        finally:
            self.get_logger().info('Action finished, proceeding next action.')
            self.execute_next_action() #？
    
    def quaternion_to_euler(self, qw, qx, qy, qz):
        """将四元数转换为欧拉角（获取 yaw 角）"""
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def low_state_handler(self, msg):
        """从 LowState 获取 IMU 四元数并计算朝向"""
        try:
            if hasattr(msg, 'imu_state') and hasattr(msg.imu_state, 'quaternion'):
                qw = msg.imu_state.quaternion[0]  
                qx = msg.imu_state.quaternion[1]  
                qy = msg.imu_state.quaternion[2]  
                qz = msg.imu_state.quaternion[3]  
                
                self.current_theta = self.quaternion_to_euler(qw, qx, qy, qz)
                
                if hasattr(self, '_last_theta_log_time'):
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    if current_time - self._last_theta_log_time > 2.0: 
                        self.get_logger().info(f'Robot orientation: θ={self.current_theta:.3f} rad ({math.degrees(self.current_theta):.1f} deg)')
                        self._last_theta_log_time = current_time
                else:
                    self._last_theta_log_time = self.get_clock().now().nanoseconds / 1e9
            else:
                if abs(self.current_vx) > 0.01 or abs(self.current_vy) > 0.01:
                    self.current_theta = math.atan2(self.current_vy, self.current_vx)
        except Exception as e:
            self.get_logger().error(f'Error processing LowState: {e}')
            
    def high_state_handler(self, msg):
        self.current_x = msg.position[0]  
        self.current_y = msg.position[1]  
        self.current_z = msg.position[2]  
    
        self.current_vx = msg.velocity[0]  
        self.current_vy = msg.velocity[1]  
        self.current_vz = msg.velocity[2]  
    
        if hasattr(self, '_last_log_time'):
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_log_time > 1.0: 
                self.get_logger().info(f'Robot position: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f})')
                self._last_log_time = current_time
        else:
            self._last_log_time = self.get_clock().now().nanoseconds / 1e9
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def unicycle_controller(self):
        error_x = self.target_x - self.current_x
        error_y = self.target_y - self.current_y
        distance_error = math.sqrt(error_x**2 + error_y**2)
        
        self.get_logger().info(f'Current: ({self.current_x:.2f}, {self.current_y:.2f}, θ={self.current_theta:.2f}), '
                              f'Target: ({self.target_x:.2f}, {self.target_y:.2f}), '
                              f'Distance: {distance_error:.2f}m')

        if distance_error < self.tolerance:
            self.get_logger().info('Reached target position.')
            self.stop_robot()
            self.execute_next_action()
            return

        target_theta = math.atan2(error_y, error_x)
        angle_error = self.normalize_angle(target_theta - self.current_theta)
        
        angle_threshold = 0.4  
        
        self.get_logger().info(f'Target angle: {target_theta:.2f}, Current angle: {self.current_theta:.2f}, '
                              f'Angle error: {angle_error:.2f} rad')
        
        if abs(angle_error) > angle_threshold:
            if angle_error > 0:
                cmd_msg = String()
                cmd_msg.data = "turn_left"
                self.cmd_publisher.publish(cmd_msg)
                self.get_logger().info('Turning left to correct orientation')
            else:
                cmd_msg = String()
                cmd_msg.data = "turn_right"
                self.cmd_publisher.publish(cmd_msg)
                self.get_logger().info('Turning right to correct orientation')
        else:
            cmd_msg = String()
            cmd_msg.data = "walk"
            self.cmd_publisher.publish(cmd_msg)
            self.get_logger().info('Walking forward to target')

    def stop_robot(self):
        """停止机器人运动"""
        cmd_msg = String()
        cmd_msg.data = "stop"
        self.cmd_publisher.publish(cmd_msg)
        
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
            
        self.state = NodeState.IDLE
        self.get_logger().info('Robot stopped and state set to IDLE.')
    
    def send_navigation_goal(self, x, y):
        self.target_x = x
        self.target_y = y
        
        self.get_logger().info(f'Starting custom navigation to ({x}, {y})')
        
        self.control_timer = self.create_timer(0.5, self.unicycle_controller)

def main(args=None):
    rclpy.init(args=args)
    task_executor_node=TaskExecutorNode()
    rclpy.spin(task_executor_node)
    task_executor_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()