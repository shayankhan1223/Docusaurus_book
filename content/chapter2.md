# Chapter 2: ROS 2 Fundamentals for Humanoid Robots

## 1. Conceptual Foundation

Robot Operating System 2 (ROS 2) is a middleware framework designed specifically for robotics applications. Unlike traditional operating systems, ROS 2 provides communication protocols, hardware abstraction, device drivers, and libraries that enable complex robotic systems to be built from modular components.

For humanoid robots, ROS 2 serves as the backbone that connects perception systems (cameras, sensors), control systems (joint controllers, balance algorithms), and high-level decision-making components. This distributed architecture is essential for humanoid robots due to their complexity, with dozens of joints, multiple sensors, and intricate control requirements.

ROS 2 addresses key challenges in humanoid robotics:
- **Real-time communication**: Ensuring timely exchange of sensor data and control commands
- **Hardware abstraction**: Allowing the same algorithms to run on different physical platforms
- **Component modularity**: Enabling parallel development of different robot subsystems
- **Safety mechanisms**: Providing tools for safe robot operation and emergency handling

## 2. Core Theory

### 2.1 Communication Architecture

ROS 2 uses a publish-subscribe model with additional services and action interfaces:

**Topics (Publish-Subscribe)**: Asynchronous communication for continuous data streams like sensor readings or motor commands. Publishers send messages to topics, and subscribers receive them without direct connection.

**Services**: Synchronous request-response communication for discrete operations like requesting a specific action or querying system status.

**Actions**: Asynchronous communication with feedback for long-running operations like navigation goals or manipulation tasks.

### 2.2 Quality of Service (QoS) Policies

ROS 2 introduces QoS policies to handle real-time requirements critical for humanoid robots:
- **Reliability**: Whether messages must be delivered (reliable) or can be dropped (best-effort)
- **Durability**: Whether late-joining subscribers receive old messages (transient-local) or only new ones (volatile)
- **History**: How many messages to store for late subscribers
- **Deadline**: Maximum time allowed for message delivery

### 2.3 Real-time Considerations for Humanoids

Humanoid robots require different QoS configurations for different subsystems:
- **High-frequency control** (1000+ Hz): Best-effort, minimal history, strict deadlines
- **Perception data** (30-60 Hz): Reliable, moderate history, reasonable deadlines
- **High-level commands** (1-10 Hz): Reliable, persistent, longer deadlines

## 3. Practical Tooling

### 3.1 Core ROS 2 Tools
- **ros2 run**: Execute ROS 2 nodes
- **ros2 topic**: Monitor and interact with topics
- **ros2 service**: Call services and check service status
- **ros2 action**: Interact with action servers
- **rviz2**: 3D visualization for robot state and sensor data
- **rqt**: GUI-based monitoring and debugging tools

### 3.2 Build Systems
- **ament_cmake**: CMake-based build system
- **colcon**: Multi-package build tool
- **rosdep**: Dependency management system

### 3.3 Simulation Integration
- **Gazebo Bridge**: Connect ROS 2 with Gazebo simulation
- **Ignition Transport**: Communication layer for simulation
- **Robot Description Format (URDF/XACRO)**: Robot model definition

## 4. Implementation Walkthrough

Let's build a complete ROS 2 system for a humanoid robot's walking controller:

### 4.1 Package Structure
```
humanoid_control/
├── CMakeLists.txt
├── package.xml
├── src/
│   ├── walking_controller.cpp
│   └── balance_node.cpp
├── include/
│   └── humanoid_control/
│       ├── walking_controller.hpp
│       └── balance_controller.hpp
├── config/
│   └── humanoid_params.yaml
└── launch/
    └── humanoid_system.launch.py
```

### 4.2 Core Implementation

```python
# walking_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from humanoid_msgs.msg import StepCommand, BalanceState
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.balance_state_pub = self.create_publisher(BalanceState, '/balance_state', 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.step_cmd_sub = self.create_subscription(StepCommand, '/step_command', self.step_command_callback, 10)

        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_height = 0.05  # meters
        self.step_duration = 1.0  # seconds
        self.com_height = 0.8  # center of mass height

        # Robot state
        self.current_joint_states = JointState()
        self.imu_data = Imu()
        self.support_foot = 'left'  # left or right
        self.walking_phase = 0.0  # 0.0 to 1.0
        self.target_velocity = Twist()

        # Walking pattern generator
        self.trajectory_generator = WalkingTrajectoryGenerator(
            step_length=self.step_length,
            step_height=self.step_height,
            com_height=self.com_height
        )

        # Control loop at 100Hz (10ms)
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Walking Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance feedback"""
        self.imu_data = msg
        # Extract orientation and angular velocity
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.robot_orientation = R.from_quat(quat)
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.current_joint_states = msg

    def step_command_callback(self, msg):
        """Handle step commands from higher-level planner"""
        self.target_velocity.linear.x = msg.velocity_x
        self.target_velocity.angular.z = msg.turn_rate
        self.step_length = msg.step_length

    def generate_walking_trajectory(self):
        """Generate foot trajectory for current walking phase"""
        # Calculate target foot position based on walking phase
        phase = self.walking_phase
        if phase < 0.5:  # Support phase
            # Maintain current support foot position
            support_foot_pos = self.get_current_support_foot_position()
            swing_foot_pos = self.calculate_swing_trajectory(phase * 2.0)
        else:  # Swing phase
            # Swap support and swing feet
            support_foot_pos = self.calculate_swing_trajectory((phase - 0.5) * 2.0)
            swing_foot_pos = self.get_current_support_foot_position()

        return support_foot_pos, swing_foot_pos

    def calculate_swing_trajectory(self, normalized_time):
        """Calculate swing foot trajectory using 5th order polynomial"""
        # 5th order polynomial for smooth foot trajectory
        # From: https://www.researchgate.net/publication/224198655
        t = normalized_time  # 0 to 1
        swing_height = self.step_height

        # 5th order polynomial coefficients for smooth trajectory
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 10.0
        a4 = -15.0
        a5 = 6.0

        x_progress = a3 * t**3 + a4 * t**4 + a5 * t**5
        z_height = 5.0 * t**2 - 10.0 * t**3 + 5.0 * t**4  # Parabolic height profile

        return np.array([x_progress * self.step_length, 0.0, z_height * swing_height])

    def balance_control(self):
        """Implement balance control using IMU feedback"""
        # Simple balance controller using IMU data
        roll, pitch, yaw = self.robot_orientation.as_euler('xyz')

        # PID-like balance control
        roll_error = -roll  # Want to maintain upright
        pitch_error = -pitch

        # Apply corrective torques to maintain balance
        # This would interface with joint controllers in practice
        balance_torques = {
            'left_hip_roll': roll_error * 0.5,
            'right_hip_roll': -roll_error * 0.5,
            'left_hip_pitch': pitch_error * 0.3,
            'right_hip_pitch': pitch_error * 0.3
        }

        return balance_torques

    def control_loop(self):
        """Main control loop running at 100Hz"""
        # Update walking phase based on target velocity
        if self.target_velocity.linear.x > 0.001:  # If moving forward
            self.walking_phase += 0.01 / self.step_duration  # 100Hz * step duration
            if self.walking_phase >= 1.0:
                self.walking_phase = 0.0
                # Switch support foot
                self.support_foot = 'right' if self.support_foot == 'left' else 'left'

        # Generate walking trajectory
        support_pos, swing_pos = self.generate_walking_trajectory()

        # Calculate balance corrections
        balance_torques = self.balance_control()

        # Generate joint commands
        joint_commands = self.calculate_joint_commands(support_pos, swing_pos, balance_torques)

        # Publish joint commands
        self.joint_cmd_pub.publish(joint_commands)

        # Publish balance state
        balance_state = BalanceState()
        balance_state.support_foot = self.support_foot
        balance_state.phase = self.walking_phase
        balance_state.com_position = self.calculate_com_position()
        self.balance_state_pub.publish(balance_state)

    def calculate_joint_commands(self, support_pos, swing_pos, balance_torques):
        """Convert desired foot positions to joint commands"""
        # Inverse kinematics to convert foot positions to joint angles
        # This is a simplified representation - real IK would be more complex

        joint_cmd = JointState()
        joint_cmd.name = [
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]
        joint_cmd.position = [0.0] * 12  # Placeholder - would use actual IK

        # Apply balance corrections
        for i, joint_name in enumerate(joint_cmd.name):
            if joint_name in balance_torques:
                # Convert balance torques to position offsets
                joint_cmd.position[i] += balance_torques[joint_name] * 0.01  # Small offset

        return joint_cmd

    def calculate_com_position(self):
        """Calculate center of mass position (simplified)"""
        # This would use forward kinematics in a real implementation
        return np.array([0.0, 0.0, self.com_height])

    def get_current_support_foot_position(self):
        """Get current position of support foot"""
        # This would query current joint states and calculate foot position
        return np.array([0.0, 0.0, 0.0])

class WalkingTrajectoryGenerator:
    """Generate walking trajectories using various methods"""

    def __init__(self, step_length=0.3, step_height=0.05, com_height=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.com_height = com_height

    def generate_com_trajectory(self, phase, velocity):
        """Generate center of mass trajectory using inverted pendulum model"""
        # Simplified inverted pendulum model for CoM trajectory
        omega = np.sqrt(9.81 / self.com_height)  # Natural frequency
        com_x = velocity * phase * self.step_length
        com_y = 0.0  # Keep CoM over support foot
        com_z = self.com_height  # Keep constant height

        return np.array([com_x, com_y, com_z])

def main(args=None):
    rclpy.init(args=args)
    node = WalkingController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down walking controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4.3 Launch File Configuration

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Walking controller node
    walking_controller = Node(
        package='humanoid_control',
        executable='walking_controller',
        name='walking_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            os.path.join(get_package_share_directory('humanoid_control'), 'config', 'humanoid_params.yaml')
        ],
        output='screen'
    )

    # Balance controller node
    balance_controller = Node(
        package='humanoid_control',
        executable='balance_controller',
        name='balance_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Joint state controller
    joint_controller = Node(
        package='joint_state_controller',
        executable='joint_state_broadcaster',
        name='joint_state_broadcaster',
        output='screen'
    )

    return LaunchDescription([
        walking_controller,
        balance_controller,
        joint_controller
    ])
```

## 5. Simulation → Real World Mapping

### 5.1 Simulation Setup for Humanoid Robots

In simulation, ROS 2 connects to physics engines like Gazebo through the Gazebo ROS 2 bridge:

```xml
<!-- URDF with ROS 2 integration -->
<robot name="humanoid_robot">
  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <update_rate>100</update_rate>
      <joint_name>left_hip_pitch</joint_name>
      <joint_name>left_knee</joint_name>
      <!-- ... other joints ... -->
    </plugin>
  </gazebo>

  <!-- Joint controllers -->
  <gazebo>
    <plugin name="position_controllers" filename="libgazebo_ros2_control.so">
      <robot_namespace>/humanoid</robot_namespace>
    </plugin>
  </gazebo>
</robot>
```

### 5.2 Real Robot Considerations

When transitioning from simulation to real hardware:
- **Timing differences**: Real robots have communication delays and processing latencies
- **Hardware limitations**: Joint limits, motor saturation, and sensor noise
- **Safety systems**: Emergency stops and protective limits must be properly configured
- **Calibration**: Sensor offsets and mechanical tolerances need real-world calibration

## 6. Common Mistakes & Mental Models

### 6.1 Common Mistakes
- **Ignoring QoS settings**: Using default QoS for time-critical control loops
- **Blocking operations**: Performing heavy computation in callback functions
- **Memory leaks**: Not properly cleaning up subscriptions and publishers
- **Race conditions**: Accessing shared data without proper synchronization
- **Inadequate error handling**: Not handling connection failures or sensor timeouts

### 6.2 Mental Models for Success
- **Component thinking**: Treat each robot subsystem as an independent node
- **Message flow**: Visualize how data flows through the system
- **Timing awareness**: Understand the timing requirements of different components
- **Failure resilience**: Design systems that handle component failures gracefully

## 7. Mini Case Study: ROS 2 in Real Humanoid Robots

### 7.1 PAL Robotics REEM-C

PAL Robotics' REEM-C humanoid robot uses ROS 2 for its complete software stack, demonstrating several key principles:

**Modular Architecture**: The robot's software is divided into independent nodes handling perception, navigation, manipulation, and human-robot interaction. This allows different teams to work on different subsystems without interfering with each other.

**Hardware Abstraction**: The same ROS 2 interfaces work whether the robot is in simulation or on the physical platform, enabling seamless transition between development and deployment.

**Real-time Performance**: The system uses custom QoS settings and real-time scheduling to ensure critical control loops maintain their timing requirements.

### 7.2 Lessons Learned

The success of ROS 2 in humanoid robotics demonstrates that:
- **Standardized interfaces** enable rapid development and integration
- **Distributed architecture** provides fault tolerance and modularity
- **Real-time capabilities** are essential for stable robot control
- **Simulation integration** accelerates development and testing

These principles have made ROS 2 the de facto standard for humanoid robot development, providing the foundation for advanced embodied intelligence systems.