# Chapter 6: Control Systems and Actuation

## 1. Conceptual Foundation

Control systems in Physical AI serve as the nervous system that translates high-level commands into precise physical movements. For humanoid robots, control systems face unique challenges due to the high degrees of freedom, underactuation, and the need to maintain dynamic balance while performing complex tasks. Unlike traditional industrial robots that operate in structured environments, humanoid robots must adapt their control strategies to varying terrains, unexpected disturbances, and the need for compliant interaction with humans and objects.

The control architecture for humanoid robots typically involves multiple control layers operating at different frequencies: high-frequency joint-level control (1000+ Hz) for motor position and torque control, mid-frequency balance control (100-500 Hz) for maintaining stability, and low-frequency task-level control (1-10 Hz) for high-level behavior execution. This hierarchical structure enables the robot to respond to disturbances at appropriate timescales while achieving complex behaviors.

Actuation systems provide the physical means for robot movement, converting control signals into mechanical force and motion. The choice of actuators, their placement, and their control significantly impact the robot's capabilities, efficiency, and safety. For humanoid robots, actuation systems must provide sufficient torque and speed while maintaining safety through compliant control and proper mechanical design.

## 2. Core Theory

### 2.1 Control System Fundamentals

**Feedback Control**: The foundation of robot control, where sensor measurements are used to adjust control outputs to achieve desired behavior. The basic feedback control loop consists of: measurement → error calculation → control law → actuation → system response → measurement (repeat).

**PID Control**: Proportional-Integral-Derivative control provides a systematic approach to feedback control. The proportional term responds to current error, the integral term eliminates steady-state error, and the derivative term provides damping to reduce oscillations.

**State-Space Control**: Modern control theory uses state-space representations to model and control complex multi-input multi-output systems. This approach is particularly important for humanoid robots with many degrees of freedom.

### 2.2 Multi-Rate Control Architecture

Humanoid robots require control systems operating at multiple timescales:

**High-Rate Joint Control (1000+ Hz)**: Direct motor control for position, velocity, and torque. This layer handles the fastest dynamics and provides the foundation for higher-level control.

**Mid-Rate Balance Control (100-500 Hz)**: Maintains center of mass stability and posture. This layer coordinates multiple joints to maintain balance while performing tasks.

**Low-Rate Task Control (1-10 Hz)**: High-level behavior execution and task planning. This layer determines what the robot should do based on goals and environmental conditions.

### 2.3 Compliance and Impedance Control

**Impedance Control**: Controls the relationship between position and force, allowing robots to behave like springs, dampers, or more complex mechanical systems. This is crucial for safe human-robot interaction and compliant manipulation.

**Admittance Control**: Controls the relationship between force and position, useful for tasks requiring force control such as assembly or surface following.

## 3. Practical Tooling

### 3.1 ROS 2 Control Framework
- **ros2_control**: Hardware abstraction and control framework
- **Joint State Controller**: Publishes joint states
- **Position/Torque Controllers**: Low-level joint control
- **Forward Command Controller**: Command forwarding for complex controllers

### 3.2 Control Libraries
- **control_toolbox**: PID controllers and other basic control tools
- **realtime_tools**: Real-time safe control utilities
- **filters**: Signal processing and filtering tools
- **control_msgs**: Control command and feedback messages

### 3.3 Simulation Integration
- **Gazebo ROS Control**: Integration with Gazebo physics simulation
- **Ignition Control**: Control system for Ignition Gazebo
- **Hardware Interfaces**: Standardized interfaces for real hardware

## 4. Implementation Walkthrough

Let's build a complete control system for a humanoid robot:

```python
# control_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import WrenchStamped, Twist
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidControlSystem(Node):
    def __init__(self):
        super().__init__('control_system')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, '/center_of_pressure', 10)
        self.balance_state_pub = self.create_publisher(Float64MultiArray, '/balance_state', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.desired_state_sub = self.create_subscription(JointState, '/desired_states', self.desired_state_callback, 10)
        self.trajectory_sub = self.create_subscription(Float64MultiArray, '/trajectory_commands', self.trajectory_callback, 10)

        # Control components
        self.joint_controllers = {}
        self.balance_controller = BalanceController()
        self.trajectory_tracker = TrajectoryTracker()
        self.computer_torque_controller = ComputedTorqueController()

        # Robot state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_joint_efforts = {}
        self.imu_data = Imu()
        self.desired_joint_positions = {}
        self.trajectory_commands = []

        # Control parameters
        self.control_frequency = 1000.0  # Hz for joint control
        self.balance_frequency = 500.0   # Hz for balance control
        self.trajectory_frequency = 100.0 # Hz for trajectory tracking

        # Initialize joint controllers
        self.initialize_joint_controllers()

        # Control timers
        self.joint_control_timer = self.create_timer(1.0/self.control_frequency, self.joint_control_loop)
        self.balance_control_timer = self.create_timer(1.0/self.balance_frequency, self.balance_control_loop)
        self.trajectory_control_timer = self.create_timer(1.0/self.trajectory_frequency, self.trajectory_control_loop)

        self.get_logger().info('Control system initialized')

    def initialize_joint_controllers(self):
        """Initialize joint controllers for all robot joints"""
        # Define humanoid robot joint names
        joint_names = [
            'left_hip_roll', 'left_hip_pitch', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_roll', 'right_hip_pitch', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow'
        ]

        # Initialize PID controllers for each joint
        for joint_name in joint_names:
            self.joint_controllers[joint_name] = PIDController(
                kp=100.0,  # Proportional gain
                ki=10.0,   # Integral gain
                kd=10.0,   # Derivative gain
                max_output=100.0  # Maximum torque (Nm)
            )

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_joint_efforts[name] = msg.effort[i]

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg

    def desired_state_callback(self, msg):
        """Update desired joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.desired_joint_positions[name] = msg.position[i]

    def trajectory_callback(self, msg):
        """Update trajectory commands"""
        self.trajectory_commands = list(msg.data)

    def joint_control_loop(self):
        """High-frequency joint control loop"""
        # Calculate control commands for each joint
        joint_commands = JointState()
        joint_commands.header.stamp = self.get_clock().now().to_msg()
        joint_commands.name = list(self.joint_controllers.keys())

        for joint_name in joint_commands.name:
            # Get current and desired states
            current_pos = self.current_joint_positions.get(joint_name, 0.0)
            current_vel = self.current_joint_velocities.get(joint_name, 0.0)
            desired_pos = self.desired_joint_positions.get(joint_name, current_pos)

            # Calculate control command using PID
            control_output = self.joint_controllers[joint_name].update(
                desired_pos, current_pos, current_vel
            )

            # Add to commands
            joint_commands.position.append(desired_pos)
            joint_commands.effort.append(control_output)

        # Publish joint commands
        self.joint_cmd_pub.publish(joint_commands)

    def balance_control_loop(self):
        """Mid-frequency balance control loop"""
        # Calculate center of mass position and velocity
        com_position = self.calculate_com_position()
        com_velocity = self.calculate_com_velocity()

        # Get support polygon from contact points
        support_polygon = self.calculate_support_polygon()

        # Calculate balance control commands
        balance_commands = self.balance_controller.calculate_balance_control(
            com_position, com_velocity, support_polygon, self.imu_data
        )

        # Apply balance control to joint commands
        self.apply_balance_control(balance_commands)

        # Publish balance state
        balance_state_msg = Float64MultiArray()
        balance_state_msg.data = [
            com_position[0], com_position[1], com_position[2],  # CoM position
            com_velocity[0], com_velocity[1], com_velocity[2],  # CoM velocity
            balance_commands[0], balance_commands[1]           # Balance corrections
        ]
        self.balance_state_pub.publish(balance_state_msg)

    def trajectory_control_loop(self):
        """Low-frequency trajectory tracking control"""
        if len(self.trajectory_commands) > 0:
            # Follow trajectory using computed torque control
            torque_commands = self.computer_torque_controller.calculate_trajectory_control(
                self.trajectory_commands,
                self.current_joint_positions,
                self.current_joint_velocities
            )

            # Apply trajectory control
            self.apply_trajectory_control(torque_commands)

    def calculate_com_position(self):
        """Calculate center of mass position (simplified)"""
        # In a real implementation, this would use forward kinematics and link masses
        # For this example, we'll estimate based on joint positions
        com_x = 0.0
        com_y = 0.0
        com_z = 0.8  # Approximate CoM height for humanoid

        # Add contributions from each joint based on its position and estimated mass
        for joint_name, position in self.current_joint_positions.items():
            # Simplified mass distribution model
            if 'hip' in joint_name:
                com_z += 0.1 * math.sin(position)  # Hip contribution
            elif 'knee' in joint_name:
                com_z -= 0.05 * math.cos(position)  # Knee contribution

        return np.array([com_x, com_y, com_z])

    def calculate_com_velocity(self):
        """Calculate center of mass velocity (simplified)"""
        # In a real implementation, this would differentiate CoM position
        # For this example, we'll estimate based on joint velocities
        com_vx = 0.0
        com_vy = 0.0
        com_vz = 0.0

        # Simplified velocity calculation
        for joint_name, velocity in self.current_joint_velocities.items():
            if 'hip' in joint_name:
                com_vz += 0.05 * velocity
            elif 'ankle' in joint_name:
                com_vx += 0.02 * velocity

        return np.array([com_vx, com_vy, com_vz])

    def calculate_support_polygon(self):
        """Calculate support polygon from contact points"""
        # For a bipedal robot, support polygon is defined by foot contact points
        # This would use force/torque sensors or contact detection in a real system
        left_foot_pos = np.array([0.1, 0.1, 0.0])  # Example position
        right_foot_pos = np.array([-0.1, 0.1, 0.0])  # Example position

        # Define support polygon vertices
        vertices = [
            left_foot_pos[:2],   # Left foot center
            right_foot_pos[:2],  # Right foot center
            # Add corners of feet for more accurate support polygon
            left_foot_pos[:2] + np.array([0.1, 0.05]),   # Left foot front right
            left_foot_pos[:2] + np.array([0.1, -0.05]),  # Left foot front left
            left_foot_pos[:2] + np.array([-0.1, 0.05]),  # Left foot back right
            left_foot_pos[:2] + np.array([-0.1, -0.05]), # Left foot back left
            right_foot_pos[:2] + np.array([0.1, 0.05]),  # Right foot front right
            right_foot_pos[:2] + np.array([0.1, -0.05]), # Right foot front left
            right_foot_pos[:2] + np.array([-0.1, 0.05]), # Right foot back right
            right_foot_pos[:2] + np.array([-0.1, -0.05]) # Right foot back left
        ]

        return np.array(vertices)

    def apply_balance_control(self, balance_commands):
        """Apply balance control commands to joint controllers"""
        # Balance commands modify desired joint positions to maintain stability
        for joint_name in self.joint_controllers.keys():
            # Apply small corrections to maintain balance
            correction = 0.0
            if 'hip' in joint_name:
                correction = balance_commands[0] * 0.1  # Hip correction
            elif 'ankle' in joint_name:
                correction = balance_commands[1] * 0.05  # Ankle correction

            if joint_name in self.desired_joint_positions:
                self.desired_joint_positions[joint_name] += correction

    def apply_trajectory_control(self, torque_commands):
        """Apply trajectory control commands"""
        # In a real system, this would convert torque commands to joint commands
        # For this example, we'll modify the joint controllers' desired positions
        for i, joint_name in enumerate(self.joint_controllers.keys()):
            if i < len(torque_commands):
                # Convert torque command to position offset
                position_offset = torque_commands[i] * 0.001  # Simplified conversion
                if joint_name in self.desired_joint_positions:
                    self.desired_joint_positions[joint_name] += position_offset

class PIDController:
    """PID controller implementation"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update(self, desired, actual, actual_velocity=None):
        """Update PID controller and return control output"""
        current_time = self.get_node().get_clock().now().nanoseconds * 1e-9

        if self.last_time is None:
            self.last_time = current_time
            return 0.0

        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        error = desired - actual

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term (use velocity if available, otherwise estimate)
        if actual_velocity is not None:
            derivative = -actual_velocity
        else:
            derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output to prevent windup
        output = max(-self.max_output, min(self.max_output, output))

        # Store values for next iteration
        self.previous_error = error
        self.last_time = current_time

        return output

class BalanceController:
    """Balance controller for humanoid robots"""

    def __init__(self):
        self.com_height = 0.8  # Center of mass height (meters)
        self.gravity = 9.81    # Gravity (m/s^2)
        self.zmp_margin = 0.05 # Safety margin for ZMP (meters)

    def calculate_balance_control(self, com_position, com_velocity, support_polygon, imu_data):
        """Calculate balance control commands using ZMP (Zero Moment Point)"""
        # Calculate ZMP (Zero Moment Point) - point where net moment is zero
        # ZMP_x = CoM_x - (CoM_height / gravity) * CoM_acceleration_x
        # ZMP_y = CoM_y - (CoM_height / gravity) * CoM_acceleration_y

        # Estimate CoM acceleration from velocity (simplified)
        # In a real system, this would use IMU data or more sophisticated estimation
        com_acceleration = np.array([0.0, 0.0, 0.0])

        # Extract orientation from IMU for gravity compensation
        quat = [imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w]
        rotation = R.from_quat(quat)

        # Calculate desired ZMP based on CoM position and velocity
        zmp_desired_x = com_position[0] - (self.com_height / self.gravity) * com_acceleration[0]
        zmp_desired_y = com_position[1] - (self.com_height / self.gravity) * com_acceleration[1]

        # Calculate current ZMP from support polygon center (simplified)
        if len(support_polygon) > 0:
            support_center = np.mean(support_polygon, axis=0)
            zmp_current_x = support_center[0]
            zmp_current_y = support_center[1]
        else:
            zmp_current_x = 0.0
            zmp_current_y = 0.0

        # Calculate error between desired and current ZMP
        zmp_error_x = zmp_desired_x - zmp_current_x
        zmp_error_y = zmp_desired_y - zmp_current_y

        # Check if ZMP is within support polygon (simplified check)
        zmp_in_support = self.is_point_in_polygon(
            np.array([zmp_desired_x, zmp_desired_y]),
            support_polygon
        )

        # Calculate balance corrections
        if not zmp_in_support:
            # If ZMP is outside support, apply strong corrections
            balance_x = zmp_error_x * 2.0  # Higher gain for instability
            balance_y = zmp_error_y * 2.0
        else:
            # If stable, apply smaller corrections
            balance_x = zmp_error_x * 1.0
            balance_y = zmp_error_y * 1.0

        # Limit corrections to prevent excessive movements
        balance_x = max(-0.1, min(0.1, balance_x))
        balance_y = max(-0.1, min(0.1, balance_y))

        return np.array([balance_x, balance_y])

    def is_point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

class TrajectoryTracker:
    """Trajectory tracking controller"""

    def __init__(self):
        self.position_tolerance = 0.01  # meters
        self.velocity_tolerance = 0.1   # m/s

    def track_trajectory(self, current_state, desired_trajectory, time):
        """Track desired trajectory"""
        # Find closest trajectory point
        if len(desired_trajectory) == 0:
            return np.array([0.0, 0.0, 0.0])  # No movement

        # Interpolate trajectory at current time
        target_state = self.interpolate_trajectory(desired_trajectory, time)

        # Calculate tracking error
        position_error = target_state[:3] - current_state[:3]
        velocity_error = target_state[3:6] - current_state[3:6]

        # Calculate control command
        kp_pos = 10.0  # Position gain
        kp_vel = 2.0   # Velocity gain

        control_command = kp_pos * position_error + kp_vel * velocity_error

        return control_command

    def interpolate_trajectory(self, trajectory, time):
        """Interpolate trajectory at given time"""
        # Simplified interpolation - in practice, this would use splines or other methods
        if len(trajectory) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # For this example, return the first trajectory point
        # In a real system, this would properly interpolate based on time
        return trajectory[0] if len(trajectory) > 0 else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

class ComputedTorqueController:
    """Computed torque controller for trajectory following"""

    def __init__(self):
        # Robot parameters (simplified model)
        self.mass = 50.0  # kg
        self.inertia = 10.0  # kg*m^2

    def calculate_trajectory_control(self, trajectory_commands, current_positions, current_velocities):
        """Calculate computed torque control commands"""
        # This would implement inverse dynamics for the actual robot model
        # For this example, we'll return simplified torque commands

        torque_commands = []

        # Convert trajectory commands to joint torques
        # This is a simplified example - real implementation would use robot dynamics
        for joint_name, current_pos in current_positions.items():
            if joint_name in current_velocities:
                current_vel = current_velocities[joint_name]

                # Calculate desired acceleration to follow trajectory
                desired_accel = 0.0  # This would come from trajectory
                current_accel = 0.0  # This would be estimated or measured

                # Computed torque: tau = M(q)*ddot_q_desired + C(q,dot_q)*dot_q + g(q)
                # Simplified as: tau = mass * desired_acceleration + damping * velocity
                torque = self.mass * desired_accel + 0.1 * current_vel

                torque_commands.append(torque)
            else:
                torque_commands.append(0.0)

        return torque_commands

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidControlSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down control system')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Simulation → Real World Mapping

### 5.1 Control System Simulation

```python
# control_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ControlSystemSimulator:
    def __init__(self):
        self.time = 0.0
        self.dt = 0.001  # 1ms timestep for high-frequency control
        self.joint_positions = np.zeros(12)  # 12 DOF for legs
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)
        self.com_position = np.array([0.0, 0.0, 0.8])
        self.com_velocity = np.array([0.0, 0.0, 0.0])

        # Control parameters
        self.kp = 100.0
        self.kd = 10.0
        self.ki = 1.0

        # Simulation parameters
        self.mass = 50.0
        self.gravity = 9.81

        # Data storage for plotting
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.torque_history = []

    def simulate_step(self, desired_positions):
        """Simulate one control step"""
        # Calculate control torques
        position_error = desired_positions - self.joint_positions
        velocity_error = -self.joint_velocities  # Assuming desired velocity is 0

        # PID control
        p_term = self.kp * position_error
        d_term = self.kd * velocity_error
        # Integral term would accumulate over time in real system

        self.joint_torques = p_term + d_term

        # Apply joint torques (simplified dynamics)
        joint_accelerations = self.joint_torques / 1.0  # Simplified inertia

        # Update joint velocities and positions
        self.joint_velocities += joint_accelerations * self.dt
        self.joint_positions += self.joint_velocities * self.dt

        # Update center of mass based on joint positions (simplified)
        self.update_com_position()

        # Store data for plotting
        self.time_history.append(self.time)
        self.position_history.append(self.joint_positions.copy())
        self.velocity_history.append(self.joint_velocities.copy())
        self.torque_history.append(self.joint_torques.copy())

        self.time += self.dt

    def update_com_position(self):
        """Update center of mass position based on joint positions"""
        # Simplified CoM calculation based on joint positions
        # In a real system, this would use forward kinematics
        self.com_position[2] = 0.8 + 0.1 * np.sin(self.time * 2)  # Simple oscillation

    def plot_results(self):
        """Plot simulation results"""
        if len(self.position_history) == 0:
            return

        positions = np.array(self.position_history)
        velocities = np.array(self.velocity_history)
        torques = np.array(self.torque_history)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot joint positions
        for i in range(min(6, positions.shape[1])):  # Plot first 6 joints
            ax1.plot(self.time_history, positions[:, i], label=f'Joint {i+1}')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Joint Positions Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot joint velocities
        for i in range(min(6, velocities.shape[1])):
            ax2.plot(self.time_history, velocities[:, i], label=f'Joint {i+1}')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.set_title('Joint Velocities Over Time')
        ax2.legend()
        ax2.grid(True)

        # Plot joint torques
        for i in range(min(6, torques.shape[1])):
            ax3.plot(self.time_history, torques[:, i], label=f'Joint {i+1}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (Nm)')
        ax3.set_title('Joint Torques Over Time')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

def run_control_simulation():
    """Run control system simulation"""
    simulator = ControlSystemSimulator()

    # Simulate for 5 seconds
    for t in np.arange(0, 5, simulator.dt):
        # Generate desired positions (square wave for testing)
        desired_positions = np.sin(t * 2) * 0.5  # Simple oscillating command
        desired_positions = np.full(12, desired_positions)  # Apply to all joints

        simulator.simulate_step(desired_positions)

    simulator.plot_results()

if __name__ == "__main__":
    run_control_simulation()
```

### 5.2 Real-World Control Considerations

**Hardware Limitations**: Real actuators have torque, speed, and power limitations that must be respected.

**Safety Systems**: Emergency stops, torque limits, and position limits must be implemented.

**Calibration**: Joint offsets, gear ratios, and other parameters must be calibrated.

**Latency**: Communication and processing delays affect control performance.

## 6. Common Mistakes & Mental Models

### 6.1 Common Mistakes
- **Integral windup**: Not limiting integral terms in PID controllers
- **Ignoring actuator limits**: Requesting torques or speeds beyond actuator capabilities
- **Poor tuning**: Using generic parameters instead of robot-specific tuning
- **Inadequate safety**: Not implementing proper safety limits and emergency stops
- **Timing violations**: Not meeting real-time control deadlines

### 6.2 Mental Models for Success
- **Multi-rate thinking**: Understand the different timescales of control loops
- **Safety first**: Always design with safety limits and fail-safes
- **Systematic tuning**: Use proper system identification and tuning methods
- **Real-time awareness**: Design control systems that meet timing requirements

## 7. Mini Case Study: Control Systems in Real Humanoid Robots

### 7.1 Honda P3 Control Architecture

Honda's P3 humanoid robot demonstrated advanced control capabilities:

**Hierarchical Control**: Multiple control layers operating at different frequencies for joint control, balance, and high-level behaviors.

**Adaptive Control**: Control parameters that adapt based on terrain and task requirements.

**Compliance Control**: Safe interaction with humans through compliant actuation.

### 7.2 Technical Implementation

P3's control system featured:
- **High-frequency joint control**: 2000 Hz control for precise joint positioning
- **Balance control**: Real-time balance maintenance using ZMP control
- **Walking pattern generation**: Online gait adaptation based on sensor feedback
- **Force control**: Compliant interaction through force feedback control

### 7.3 Lessons Learned

The development of control systems for humanoid robots shows that:
- **Hierarchical control** is essential for managing complexity
- **Real-time performance** requires careful system design and optimization
- **Safety systems** must be integrated throughout the control architecture
- **Adaptive control** allows robots to handle varying conditions

These principles continue to guide the development of control systems for modern humanoid robots, emphasizing the need for robust, safe, and adaptive control architectures.