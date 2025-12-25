# Chapter 5: Planning and Navigation in Physical Space

## 1. Conceptual Foundation

Planning and navigation in Physical AI represent the cognitive layer that translates high-level goals into executable actions in three-dimensional space. Unlike traditional path planning for wheeled robots, humanoid navigation must account for complex kinematics, dynamic balance, multi-step planning, and the need to maintain stability while moving through the environment.

The planning process in Physical AI involves multiple interconnected components: global path planning to find routes to goals, local planning to avoid obstacles in real-time, motion planning to generate dynamically feasible trajectories, and balance planning to maintain stability during movement. For humanoid robots, these systems must work together seamlessly to enable safe, efficient, and human-like navigation.

Navigation in physical space adds complexity beyond traditional robotics due to the need for 3D planning, dynamic obstacle avoidance, and the integration of perception and planning systems. The robot must continuously update its understanding of the environment and replan as new information becomes available, all while maintaining real-time performance and safety.

## 2. Core Theory

### 2.1 Multi-Layer Planning Architecture

Physical AI navigation typically employs a hierarchical planning approach:

**Global Planning**: High-level route planning using topological or grid-based maps to find optimal paths from start to goal. This operates at a lower frequency (1-10 Hz) and considers static obstacles and overall route efficiency.

**Local Planning**: Short-term trajectory planning that considers immediate obstacles and dynamic conditions. This operates at higher frequency (10-50 Hz) and generates collision-free paths for the immediate future.

**Motion Planning**: Low-level trajectory generation that considers robot dynamics, joint limits, and balance constraints. This operates at the highest frequency (100+ Hz) and generates dynamically feasible movements.

### 2.2 Configuration Space and Path Planning

The configuration space (C-space) represents all possible robot configurations, where each point corresponds to a specific arrangement of the robot's joints and position. For humanoid robots, the C-space is high-dimensional due to multiple degrees of freedom.

**RRT (Rapidly-exploring Random Trees)**: Probabilistically complete planning algorithm that builds a tree of feasible configurations by randomly sampling the C-space.

**PRM (Probabilistic Roadmap)**: Pre-computes a roadmap of the C-space by sampling configurations and connecting them with collision-free paths.

**A* and Dijkstra**: Graph-based algorithms for finding optimal paths in discretized spaces.

### 2.3 Dynamic Movement Primitives (DMPs)

For humanoid robots, movement planning often uses DMPs to generate smooth, stable trajectories that can be modulated in real-time. DMPs provide a mathematical framework for learning and reproducing movements while maintaining stability properties.

## 3. Practical Tooling

### 3.1 Navigation 2 (Nav2)
- **Global Planner**: A*, Dijkstra, and other graph-based planners
- **Local Planner**: DWA, TEB, and other trajectory optimization methods
- **Costmap 2D**: 2D costmap for obstacle representation
- **Behavior Trees**: Task-level planning and execution

### 3.2 Motion Planning Libraries
- **MoveIt**: Motion planning, inverse kinematics, and trajectory generation
- **OMPL**: Open Motion Planning Library with various planning algorithms
- **CHOMP**: Covariant Hamiltonian Optimization for Motion Planning
- **STOMP**: Stochastic Trajectory Optimization for Motion Planning

### 3.3 Simulation and Testing Tools
- **Gazebo Navigation**: Navigation testing in simulation
- **RViz**: Visualization and debugging tools
- **Navigation Tuning**: Parameter optimization tools

## 4. Implementation Walkthrough

Let's build a complete navigation system for a humanoid robot:

```python
# navigation_system.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point, Vector3
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import heapq
from collections import deque

class HumanoidNavigationSystem(Node):
    def __init__(self):
        super().__init__('navigation_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_path_pub = self.create_publisher(Path, '/global_path', 10)
        self.local_path_pub = self.create_publisher(Path, '/local_path', 10)
        self.trajectory_pub = self.create_publisher(Path, '/trajectory', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/navigation_markers', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Navigation components
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()
        self.motion_planner = MotionPlanner()
        self.trajectory_tracker = TrajectoryTracker()

        # Robot state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_velocity = np.array([0.0, 0.0, 0.0])  # linear_x, linear_y, angular_z
        self.global_path = []
        self.local_path = []
        self.current_goal = None
        self.navigation_state = 'IDLE'  # IDLE, PLANNING, EXECUTING, REPLANNING

        # Navigation parameters
        self.planning_frequency = 1.0  # Hz
        self.control_frequency = 50.0  # Hz
        self.local_plan_horizon = 3.0  # meters
        self.goal_tolerance = 0.3  # meters

        # Timers
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.plan_navigation)
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.execute_navigation)

        self.get_logger().info('Navigation system initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = np.arctan2(siny_cosp, cosy_cosp)

        # Update velocity
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.linear.y
        self.current_velocity[2] = msg.twist.twist.angular.z

    def laser_callback(self, msg):
        """Process laser scan for local obstacle detection"""
        # Convert laser scan to obstacle points in robot frame
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        valid_ranges = np.array(msg.ranges)
        valid_ranges[~np.isfinite(valid_ranges)] = msg.range_max

        # Filter out far obstacles
        close_obstacles = valid_ranges < 3.0  # Only consider obstacles within 3m

        # Convert to Cartesian coordinates
        obstacle_x = valid_ranges[close_obstacles] * np.cos(angles[close_obstacles])
        obstacle_y = valid_ranges[close_obstacles] * np.sin(angles[close_obstacles])

        self.local_planner.update_obstacles(np.column_stack([obstacle_x, obstacle_y]))

    def goal_callback(self, msg):
        """Handle new navigation goal"""
        self.current_goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0  # We'll calculate orientation later
        ])
        self.navigation_state = 'PLANNING'
        self.get_logger().info(f'New goal received: {self.current_goal[:2]}')

    def plan_navigation(self):
        """Main navigation planning loop"""
        if self.navigation_state == 'PLANNING' and self.current_goal is not None:
            # Plan global path
            start = self.current_pose.copy()
            goal = self.current_goal.copy()

            # Plan global path
            self.global_path = self.global_planner.plan_path(start, goal)

            if len(self.global_path) > 0:
                # Plan local path
                self.local_path = self.local_planner.plan_local_path(
                    start, self.global_path, self.current_velocity
                )

                # Publish global path
                self.publish_path(self.global_path, self.global_path_pub, 'global_path')

                # Publish local path
                self.publish_path(self.local_path, self.local_path_pub, 'local_path')

                self.navigation_state = 'EXECUTING'
                self.get_logger().info('Global and local paths planned successfully')
            else:
                self.get_logger().warn('Failed to plan global path')
                self.navigation_state = 'IDLE'

        elif self.navigation_state == 'EXECUTING':
            # Check if we need to replan
            if self.should_replan():
                self.navigation_state = 'REPLANNING'

        elif self.navigation_state == 'REPLANNING':
            # Replan local path due to new obstacles or other conditions
            start = self.current_pose.copy()
            self.local_path = self.local_planner.plan_local_path(
                start, self.global_path, self.current_velocity
            )
            self.publish_path(self.local_path, self.local_path_pub, 'local_path')
            self.navigation_state = 'EXECUTING'

    def execute_navigation(self):
        """Execute navigation commands"""
        if self.navigation_state == 'EXECUTING' and len(self.local_path) > 0:
            # Generate control commands based on local path
            cmd_vel = self.trajectory_tracker.follow_trajectory(
                self.current_pose, self.local_path, self.current_velocity
            )

            # Check if goal reached
            if self.is_goal_reached():
                cmd_vel = Twist()  # Stop
                self.navigation_state = 'IDLE'
                self.get_logger().info('Goal reached successfully')

            self.cmd_vel_pub.publish(cmd_vel)

    def should_replan(self):
        """Check if replanning is needed"""
        # Replan if we're too far from the planned path
        if len(self.local_path) > 0:
            closest_point = self.find_closest_point(self.current_pose[:2], self.local_path)
            distance_to_path = np.linalg.norm(self.current_pose[:2] - closest_point)
            if distance_to_path > 0.5:  # 50cm threshold
                return True

        # Replan if there are new obstacles blocking the path
        if self.local_planner.has_obstacles_on_path(self.local_path, self.current_pose[:2]):
            return True

        return False

    def is_goal_reached(self):
        """Check if goal has been reached"""
        if self.current_goal is None:
            return False

        distance_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal[:2])
        return distance_to_goal <= self.goal_tolerance

    def find_closest_point(self, point, path):
        """Find the closest point on the path to the given point"""
        if len(path) == 0:
            return point

        distances = np.linalg.norm(path[:, :2] - point, axis=1)
        closest_idx = np.argmin(distances)
        return path[closest_idx, :2]

    def publish_path(self, path, publisher, frame_id):
        """Publish path as Path message"""
        if len(path) == 0:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        publisher.publish(path_msg)

class GlobalPlanner:
    """Global path planner using A* algorithm"""

    def __init__(self):
        self.grid_resolution = 0.1  # meters per cell
        self.inflation_radius = 0.5  # meters

    def plan_path(self, start, goal):
        """Plan path using A* algorithm"""
        # For this example, we'll use a simplified approach
        # In practice, this would use a proper grid map with A* or other algorithms

        # Create a simple straight-line path with intermediate waypoints
        # This would be replaced with proper A* implementation in a real system
        path = self.generate_straight_path(start, goal)

        # Add intermediate waypoints for smoother navigation
        detailed_path = self.interpolate_path(path, self.grid_resolution * 2)

        return detailed_path

    def generate_straight_path(self, start, goal):
        """Generate a straight path from start to goal"""
        # Calculate distance and direction
        direction = goal[:2] - start[:2]
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Already at goal
            return np.array([start])

        # Normalize direction
        direction = direction / distance

        # Generate path points
        num_points = int(distance / 0.2) + 1  # 20cm between points
        path = np.zeros((num_points, 3))

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            path[i, :2] = start[:2] + t * (goal[:2] - start[:2])
            path[i, 2] = np.arctan2(direction[1], direction[0])

        return path

    def interpolate_path(self, path, resolution):
        """Interpolate path to have points at specified resolution"""
        if len(path) < 2:
            return path

        # Calculate cumulative distances
        distances = [0.0]
        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i, :2] - path[i-1, :2])
            distances.append(distances[-1] + dist)

        # Create interpolation functions
        if len(path) > 1:
            fx = interp1d(distances, path[:, 0], kind='linear', fill_value='extrapolate')
            fy = interp1d(distances, path[:, 1], kind='linear', fill_value='extrapolate')

            # Generate new distances at specified resolution
            total_distance = distances[-1]
            new_distances = np.arange(0, total_distance, resolution)

            if len(new_distances) == 0:
                new_distances = [0]

            # Interpolate positions
            new_x = fx(new_distances)
            new_y = fy(new_distances)

            # Calculate orientations
            new_theta = np.zeros(len(new_x))
            for i in range(len(new_x) - 1):
                dx = new_x[i+1] - new_x[i]
                dy = new_y[i+1] - new_y[i]
                new_theta[i] = np.arctan2(dy, dx)

            if len(new_theta) > 0:
                new_theta[-1] = new_theta[-2] if len(new_theta) > 1 else 0

            detailed_path = np.column_stack([new_x, new_y, new_theta])
            return detailed_path

        return path

class LocalPlanner:
    """Local path planner for obstacle avoidance"""

    def __init__(self):
        self.horizon = 3.0  # meters
        self.time_horizon = 2.0  # seconds
        self.obstacles = np.array([]).reshape(0, 2)
        self.obstacle_inflation = 0.3  # meters

    def update_obstacles(self, obstacles):
        """Update obstacle information from sensor data"""
        self.obstacles = obstacles

    def plan_local_path(self, robot_pose, global_path, current_velocity):
        """Plan local path considering obstacles and current state"""
        if len(global_path) == 0:
            return np.array([])

        # Find current position on global path
        current_idx = self.find_current_path_index(robot_pose, global_path)

        # Extract path segment within horizon
        local_goal_idx = self.find_horizon_index(robot_pose, global_path, current_idx)
        local_path = global_path[current_idx:local_goal_idx+1]

        if len(local_path) == 0:
            # If we're near the end of the global path, create a short local path
            remaining_distance = np.linalg.norm(robot_pose[:2] - global_path[-1, :2])
            if remaining_distance < self.horizon:
                local_path = np.array([robot_pose, global_path[-1]])
            else:
                # Generate a path in the direction of the goal
                direction = global_path[-1, :2] - robot_pose[:2]
                direction = direction / np.linalg.norm(direction)
                local_path = np.array([
                    robot_pose,
                    robot_pose + np.array([direction[0], direction[1], 0]) * self.horizon
                ])

        # Apply obstacle avoidance to local path
        if len(self.obstacles) > 0:
            local_path = self.avoid_obstacles(local_path, robot_pose)

        return local_path

    def find_current_path_index(self, robot_pose, global_path):
        """Find the index on global path closest to robot position"""
        if len(global_path) == 0:
            return 0

        distances = np.linalg.norm(global_path[:, :2] - robot_pose[:2], axis=1)
        closest_idx = np.argmin(distances)

        # Look ahead to find the point that's still ahead of the robot
        for i in range(closest_idx, min(len(global_path), closest_idx + 10)):
            if self.is_ahead(robot_pose, global_path[i]):
                return i

        return closest_idx

    def is_ahead(self, robot_pose, path_point):
        """Check if path point is ahead of robot based on robot orientation"""
        direction_to_point = path_point[:2] - robot_pose[:2]
        robot_direction = np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])])
        return np.dot(direction_to_point, robot_direction) > 0

    def find_horizon_index(self, robot_pose, global_path, start_idx):
        """Find the index on global path that's within horizon distance"""
        current_distance = 0.0
        current_idx = start_idx

        for i in range(start_idx, len(global_path) - 1):
            segment_length = np.linalg.norm(
                global_path[i+1, :2] - global_path[i, :2]
            )
            if current_distance + segment_length > self.horizon:
                break
            current_distance += segment_length
            current_idx = i + 1

        return current_idx

    def avoid_obstacles(self, local_path, robot_pose):
        """Modify local path to avoid obstacles"""
        if len(local_path) < 2:
            return local_path

        # For this example, we'll implement a simple obstacle avoidance
        # In practice, this would use more sophisticated algorithms like DWA or TEB
        safe_path = [local_path[0]]

        for i in range(1, len(local_path)):
            current_point = local_path[i]
            prev_point = safe_path[-1]

            # Check if path segment has obstacles
            if not self.path_has_obstacles(prev_point, current_point):
                safe_path.append(current_point)
            else:
                # Find a detour around the obstacle
                detour_point = self.find_detour_point(prev_point, current_point)
                safe_path.append(detour_point)
                safe_path.append(current_point)

        return np.array(safe_path)

    def path_has_obstacles(self, start, end):
        """Check if path segment has obstacles"""
        # Simple check: if any obstacle is close to the path segment
        path_vector = end[:2] - start[:2]
        path_length = np.linalg.norm(path_vector)

        if path_length == 0:
            return False

        path_unit = path_vector / path_length

        for obstacle in self.obstacles:
            # Calculate distance from obstacle to line segment
            obstacle_to_start = obstacle - start[:2]
            projection = np.dot(obstacle_to_start, path_unit)
            projection = np.clip(projection, 0, path_length)

            closest_point = start[:2] + projection * path_unit
            distance = np.linalg.norm(obstacle - closest_point)

            if distance < self.obstacle_inflation:
                return True

        return False

    def find_detour_point(self, start, end):
        """Find a detour point around an obstacle"""
        # Simple detour: move perpendicular to the path
        path_vector = end[:2] - start[:2]
        path_length = np.linalg.norm(path_vector)

        if path_length == 0:
            return end

        path_unit = path_vector / path_length
        perpendicular = np.array([-path_unit[1], path_unit[0]])  # Rotate 90 degrees

        # Move perpendicular to avoid obstacle
        detour_distance = 0.5  # meters
        detour_point = (start[:2] + end[:2]) / 2 + perpendicular * detour_distance

        return np.array([detour_point[0], detour_point[1], end[2]])

    def has_obstacles_on_path(self, path, robot_position):
        """Check if there are obstacles blocking the current path"""
        if len(path) < 2 or len(self.obstacles) == 0:
            return False

        # Check each segment of the path
        for i in range(len(path) - 1):
            if self.path_has_obstacles(path[i], path[i+1]):
                return True

        return False

class MotionPlanner:
    """Motion planning for dynamically feasible trajectories"""

    def __init__(self):
        self.max_linear_vel = 0.5  # m/s
        self.max_angular_vel = 0.5  # rad/s
        self.max_linear_acc = 0.5  # m/s^2
        self.max_angular_acc = 1.0  # rad/s^2

    def plan_motion(self, local_path, current_state):
        """Plan dynamically feasible motion along local path"""
        # This would implement trajectory optimization considering robot dynamics
        # For this example, we'll return a simple velocity profile
        if len(local_path) == 0:
            return []

        # Calculate velocities along the path
        velocities = []
        for i in range(len(local_path)):
            if i == 0:
                velocities.append(np.array([0.0, 0.0, 0.0]))  # Start with zero velocity
            else:
                # Calculate desired velocity based on path direction
                direction = local_path[i, :2] - local_path[i-1, :2]
                distance = np.linalg.norm(direction)

                if distance > 0:
                    direction = direction / distance
                    linear_vel = min(self.max_linear_vel, distance * 2)  # Simple velocity scaling
                    angular_vel = local_path[i, 2] - local_path[i-1, 2]
                    angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)

                    velocities.append(np.array([linear_vel, 0.0, angular_vel]))
                else:
                    velocities.append(np.array([0.0, 0.0, 0.0]))

        return velocities

class TrajectoryTracker:
    """Track trajectory and generate control commands"""

    def __init__(self):
        self.lookahead_distance = 0.5  # meters
        self.linear_kp = 1.0  # Linear velocity proportional gain
        self.angular_kp = 2.0  # Angular velocity proportional gain
        self.path_tolerance = 0.1  # meters

    def follow_trajectory(self, current_pose, trajectory, current_velocity):
        """Follow trajectory and generate velocity commands"""
        if len(trajectory) == 0:
            cmd = Twist()
            return cmd

        # Find the closest point on trajectory
        closest_idx = self.find_closest_point_idx(current_pose, trajectory)

        # Find look-ahead point
        lookahead_idx = self.find_lookahead_point(current_pose, trajectory, closest_idx)

        if lookahead_idx is not None:
            target_point = trajectory[lookahead_idx]

            # Calculate control commands
            cmd = self.calculate_control_commands(current_pose, target_point, trajectory, lookahead_idx)
        else:
            # If no look-ahead point found, try to stop
            cmd = Twist()
            cmd.linear.x = max(-0.1, min(0.1, -current_velocity[0]))  # Gentle deceleration
            cmd.angular.z = max(-0.1, min(0.1, -current_velocity[2]))

        return cmd

    def find_closest_point_idx(self, pose, trajectory):
        """Find index of closest point on trajectory"""
        distances = np.linalg.norm(trajectory[:, :2] - pose[:2], axis=1)
        return np.argmin(distances)

    def find_lookahead_point(self, pose, trajectory, start_idx):
        """Find look-ahead point on trajectory"""
        for i in range(start_idx, len(trajectory)):
            distance = np.linalg.norm(trajectory[i, :2] - pose[:2])
            if distance >= self.lookahead_distance:
                return i

        # If no point is far enough, return the last point
        return len(trajectory) - 1 if len(trajectory) > 0 else None

    def calculate_control_commands(self, current_pose, target_point, trajectory, target_idx):
        """Calculate linear and angular velocity commands"""
        cmd = Twist()

        # Calculate position error
        error_x = target_point[0] - current_pose[0]
        error_y = target_point[1] - current_pose[1]

        # Transform error to robot frame
        cos_yaw = np.cos(-current_pose[2])
        sin_yaw = np.sin(-current_pose[2])

        local_error_x = cos_yaw * error_x - sin_yaw * error_y
        local_error_y = sin_yaw * error_x + cos_yaw * error_y

        # Calculate linear velocity (proportional to distance error)
        distance_error = np.sqrt(local_error_x**2 + local_error_y**2)
        cmd.linear.x = min(self.linear_kp * distance_error, 0.5)  # Limit speed

        # Calculate angular velocity (to correct lateral error and heading)
        heading_error = target_point[2] - current_pose[2]
        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Combine lateral error correction and heading correction
        cmd.angular.z = self.angular_kp * (np.arctan2(local_error_y, max(abs(local_error_x), 0.1)) + 0.3 * heading_error)

        # Limit angular velocity
        cmd.angular.z = max(-0.5, min(0.5, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidNavigationSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down navigation system')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Simulation → Real World Mapping

### 5.1 Navigation Simulation

```python
# navigation_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

class NavigationSimulator:
    def __init__(self):
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.robot_velocity = np.array([0.0, 0.0])
        self.goal = np.array([5.0, 5.0])
        self.obstacles = [
            {'center': np.array([2.0, 2.0]), 'radius': 0.5},
            {'center': np.array([3.5, 1.5]), 'radius': 0.3},
            {'center': np.array([1.0, 4.0]), 'radius': 0.4}
        ]
        self.path_history = []

    def simulate_step(self, cmd_vel, dt=0.02):
        """Simulate one step of robot motion"""
        # Update robot pose based on velocity commands
        self.robot_pose[0] += cmd_vel.linear.x * np.cos(self.robot_pose[2]) * dt
        self.robot_pose[1] += cmd_vel.linear.x * np.sin(self.robot_pose[2]) * dt
        self.robot_pose[2] += cmd_vel.angular.z * dt

        # Keep orientation in [-pi, pi]
        while self.robot_pose[2] > np.pi:
            self.robot_pose[2] -= 2 * np.pi
        while self.robot_pose[2] < -np.pi:
            self.robot_pose[2] += 2 * np.pi

        # Update velocity
        self.robot_velocity[0] = cmd_vel.linear.x
        self.robot_velocity[1] = cmd_vel.angular.z

        # Store path history
        self.path_history.append(self.robot_pose[:2].copy())

    def check_collision(self):
        """Check if robot collides with any obstacles"""
        robot_radius = 0.3  # Robot radius
        for obs in self.obstacles:
            distance = np.linalg.norm(self.robot_pose[:2] - obs['center'])
            if distance < (obs['radius'] + robot_radius):
                return True
        return False

    def plot_environment(self):
        """Plot the navigation environment"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot obstacles
        for obs in self.obstacles:
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.5)
            ax.add_patch(circle)

        # Plot robot path
        if len(self.path_history) > 1:
            path = np.array(self.path_history)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Robot Path')
            ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
            ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='Current')

        # Plot goal
        ax.plot(self.goal[0], self.goal[1], 'gs', markersize=12, label='Goal')

        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Humanoid Robot Navigation Simulation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def run_navigation_simulation():
    """Run a complete navigation simulation"""
    simulator = NavigationSimulator()

    # This would connect to the navigation system and run simulation
    # For this example, we'll just show the concept
    pass
```

### 5.2 Real-World Considerations

**Dynamic Obstacles**: Real environments have moving obstacles that require prediction and avoidance.

**Uncertainty Handling**: Navigation systems must handle uncertainty in localization and mapping.

**Multi-robot Coordination**: When multiple robots operate in the same space, coordination is required.

## 6. Common Mistakes & Mental Models

### 6.1 Common Mistakes
- **Local minima**: Getting stuck in local minima when using potential field methods
- **Oscillation**: Robot oscillating back and forth when trying to navigate around obstacles
- **Inadequate replanning**: Not replanning frequently enough to handle dynamic environments
- **Ignoring robot dynamics**: Planning paths that are not dynamically feasible
- **Poor parameter tuning**: Using default parameters that don't match the specific robot and environment

### 6.2 Mental Models for Success
- **Layered thinking**: Separate global planning from local planning and motion control
- **Reactive vs. predictive**: Balance reactive obstacle avoidance with predictive planning
- **Safety margins**: Always plan with adequate safety margins for uncertainty
- **Continuous improvement**: Continuously refine planning parameters based on performance

## 7. Mini Case Study: Navigation in Real Humanoid Robots

### 7.1 Honda ASIMO Navigation System

Honda's ASIMO robot demonstrated sophisticated navigation capabilities:

**Multi-modal Perception**: Combined cameras, ultrasonic sensors, and other sensors for environment awareness.

**Dynamic Obstacle Avoidance**: Could navigate around moving people and objects in real-time.

**Stair Navigation**: Could climb and descend stairs while maintaining balance.

### 7.2 Technical Implementation

ASIMO's navigation system featured:
- **Real-time mapping**: Building maps of the environment as it moved
- **Predictive algorithms**: Anticipating movements of people and objects
- **Balance integration**: Maintaining stability while navigating
- **Human-aware navigation**: Adjusting behavior based on human presence

### 7.3 Lessons Learned

The development of navigation systems for humanoid robots shows that:
- **Integration with balance** is crucial for stable locomotion
- **Real-time performance** requires efficient algorithms and proper hardware
- **Human-aware navigation** is essential for safe human-robot interaction
- **Adaptive planning** allows robots to handle dynamic environments

These insights continue to guide the development of navigation systems for modern humanoid robots, emphasizing the need for robust, safe, and human-compatible navigation capabilities.