import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
import hid
import math
import time

class HiwonderBridgeNode(Node):
    def __init__(self):
        super().__init__('hiwonder_bridge_node')
        
        # --- HID Hardware Configuration ---
        self.vid = 0x0483
        self.pid = 0x5750
        
        # 1. Map ROS joint names to Hiwonder Servo IDs
        self.joint_mapping = {
            'J1': 1,
            'J2': 2,
            'J3': 3,
            'J4': 4,
            'J5': 5
        }
        
        # 2. Hardware Home Positions (ROS 0.0 radians = These values)
        self.home_positions = {
            1: 500,
            2: 750,
            3: 750,
            4: 270,
            5: 470
        }
        
        # Initialize HID Connection
        self.board = hid.device()
        try:
            self.board.open(self.vid, self.pid)
            self.board.set_nonblocking(1) 
            self.get_logger().info(f"Successfully connected to Hiwonder HID board (VID: {self.vid:04x}).")
        except IOError as e:
            self.get_logger().error(f"Failed to open HID device: {e}")
            raise SystemExit

        # Subscribe to the arm joint states (J1-J5)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Subscribe to the Gripper commands (J6)
        self.gripper_sub = self.create_subscription(
            Int32,
            '/gripper_command',
            self.gripper_callback,
            10
        )
        
        # Throttle control (20Hz)
        self.last_write_time = time.time()
        self.write_interval = 0.05 
        
        self.get_logger().info("Hiwonder Bridge Active. Listening to /joint_states and /gripper_command.")

    def rad_to_servo_pos(self, radians, servo_id):
        home = self.home_positions.get(servo_id, 500)
        degrees = math.degrees(radians)
        units_offset = degrees * (1000.0 / 240.0)
        target_pos = int(home + units_offset)
        return max(0, min(1000, target_pos))

    def joint_state_callback(self, msg):
        current_time = time.time()
        
        if (current_time - self.last_write_time) < self.write_interval:
            return
            
        self.last_write_time = current_time
        
        packet_ids = []
        packet_positions = []
        
        for i, name in enumerate(msg.name):
            if name in self.joint_mapping:
                servo_id = self.joint_mapping[name]
                rad_pos = msg.position[i]
                hw_pos = self.rad_to_servo_pos(rad_pos, servo_id)
                
                packet_ids.append(servo_id)
                packet_positions.append(hw_pos)
                
        if packet_ids:
            move_time_ms = int(self.write_interval * 1000) 
            self.send_multi_servo_command(packet_ids, packet_positions, move_time_ms)

    def gripper_callback(self, msg):
        """
        Executes a dedicated Gripper move when commanded by dynamic_ik_tracker
        """
        pos = msg.data
        pos = max(0, min(1000, pos)) # Clamp for hardware safety
        
        self.get_logger().info(f"Executing Gripper move to position: {pos}")
        
        # Hardcoded to Servo ID 6, with a 500ms move time for clamping
        self.send_multi_servo_command([6], [pos], 500)

    def send_multi_servo_command(self, ids, positions, move_time_ms):
        num_servos = len(ids)
        data_length = 5 + (3 * num_servos)
        CMD_MULT_SERVO_MOVE = 0x03
        
        packet = [
            0x55, 0x55,
            data_length,
            CMD_MULT_SERVO_MOVE,
            num_servos,
            move_time_ms & 0xFF,
            (move_time_ms >> 8) & 0xFF
        ]
        
        for s_id, pos in zip(ids, positions):
            packet.extend([s_id, pos & 0xFF, (pos >> 8) & 0xFF])
            
        hid_report = [0x00] + packet
        while len(hid_report) < 65:
            hid_report.append(0x00)
            
        try:
            self.board.write(hid_report)
        except IOError as e:
            self.get_logger().error(f"HID write error: {e}")

def main(args=None):
    rclpy.init(args=args)
    bridge_node = HiwonderBridgeNode()
    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        bridge_node.get_logger().info("Shutting down Hiwonder Bridge.")
    finally:
        if hasattr(bridge_node, 'board'):
            bridge_node.board.close()
        bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
