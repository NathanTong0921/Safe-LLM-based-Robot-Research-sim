import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
import re
from enum import Enum, auto
import time
import math

class NodeState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    DETECTING = auto()

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger 

class TaskExecutorNode(Node):
    def __init__(self):
        super().__init__('task_executor_node')
        self.state = NodeState.IDLE
        self.actions = []
        
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
        
        self.ltl_plan_subscriber = self.create_subscription(
            String,
            'ltl_plan',
            self.ltl_plan_callback,
            10
        ) 

        # for real dog use Nav2
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.detection_service_client = self.create_client(Trigger, 'trigger_detection')

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

    def send_navigation_goal(self, x, y):
        if not self.nav_action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('Nav2 server is not available')
            self.state = NodeState.IDLE
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map' 
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0 

        self.get_logger().info(f'Sending nav target to (x={x}, y={y})...')
        self._send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg, feedback_callback=self.navigation_feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Rejected goal request by Nav2 server')
            self.state = NodeState.IDLE
            return

        self.get_logger().info('Goal accepted, executing...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Navigation feedback: {feedback.distance_remaining:.2f} meters remaining.')

    def get_navigation_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation completed, result code: {result.code}')
        self.execute_next_action() #？

    def detection_done_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Received response from detection service: "{response.message}"')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        finally:
            self.get_logger().info('Action finished, proceeding next action.')
            self.execute_next_action() #？

def main(args=None):
    rclpy.init(args=args)
    task_executor_node=TaskExecutorNode()
    rclpy.spin(task_executor_node)
    task_executor_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()