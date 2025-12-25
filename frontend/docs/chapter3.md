---
sidebar_label: "Chapter 3: Simulation Environments - Gazebo, Isaac, Unity"
sidebar_position: 377
title: "Chapter 3: Simulation Environments - Gazebo, Isaac, Unity"
---

# Chapter 3: Simulation Environments - Gazebo, Isaac, Unity

## 1. Conceptual Foundation

Simulation environments serve as the virtual laboratories for Physical AI and robotics development. They provide safe, controllable, and repeatable environments where complex robotic behaviors can be tested, refined, and validated before deployment on expensive physical hardware. For humanoid robots, simulation is particularly crucial due to the complexity of their multi-degree-of-freedom systems and the need to develop stable locomotion and manipulation capabilities.

Modern simulation environments must accurately model physics, sensor behavior, and environmental interactions to bridge the "reality gap" between virtual and physical worlds. This requires sophisticated physics engines, realistic rendering systems, and accurate sensor models that capture the noise, latency, and limitations of real hardware.

The three primary simulation platforms for Physical AI - Gazebo, Isaac Sim, and Unity - each offer unique advantages: Gazebo provides deep integration with ROS 2 and open-source accessibility, Isaac Sim offers high-fidelity graphics and AI training capabilities, and Unity provides powerful game engine features with extensive tooling for complex scenarios.

## 2. Core Theory

### 2.1 Physics Simulation Fundamentals

Simulation environments must accurately model several physical phenomena:

**Rigid Body Dynamics**: The motion of solid objects under applied forces, including translation, rotation, and collisions. This involves solving complex systems of differential equations in real-time.

**Contact Mechanics**: How objects interact when they touch, including friction, restitution (bounciness), and contact forces. These interactions are crucial for stable grasping and locomotion.

**Soft Body Dynamics**: For more advanced applications, simulating deformable objects, cloth, and fluids becomes important for realistic interaction scenarios.

### 2.2 Sensor Simulation

Realistic sensor simulation is critical for effective sim-to-real transfer:

**Camera Simulation**: Modeling lens distortion, exposure, noise, and dynamic range to match real cameras. This includes RGB, depth, and stereo vision sensors.

**LiDAR Simulation**: Replicating the scanning patterns, resolution, and noise characteristics of real LiDAR units.

**IMU Simulation**: Modeling the noise, bias, and drift characteristics of real inertial measurement units.

**Force/Torque Sensors**: Simulating the response of joint force sensors and tactile sensors.

### 2.3 Rendering and Perception

High-fidelity rendering systems enable:
- **Photorealistic simulation**: For training computer vision algorithms
- **Synthetic data generation**: Creating large datasets for AI training
- **Visual debugging**: Understanding robot behavior through visualization

## 3. Practical Tooling

### 3.1 Gazebo Harmonic
- **Physics Engine**: Open Dynamics Engine (ODE), Bullet, or DART
- **ROS 2 Integration**: Native support through Gazebo ROS 2 bridge
- **Model Database**: Extensive library of robots and environments
- **Plugin System**: Extensible through C++ plugins

### 3.2 Isaac Sim
- **Rendering Engine**: NVIDIA Omniverse for photorealistic graphics
- **AI Training**: Built-in reinforcement learning environments
- **USD Format**: Universal Scene Description for complex scenes
- **GPU Acceleration**: Leverages CUDA for high-performance simulation

### 3.3 Unity Robotics
- **Game Engine Features**: Advanced rendering and physics
- **ML-Agents**: Built-in reinforcement learning framework
- **Cross-platform**: Deploy to multiple platforms
- **Asset Store**: Extensive library of 3D models and environments

## 4. Implementation Walkthrough

### 4.1 Gazebo Setup for Humanoid Robot

Let's create a complete simulation environment for a humanoid robot in Gazebo:

```xml
<!-- humanoid_model.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_leg" value="5.0" />
  <xacro:property name="mass_torso" value="20.0" />
  <xacro:property name="mass_arm" value="3.0" />
  <xacro:property name="mass_foot" value="2.0" />

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="base_torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.8"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.15" length="0.8"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_torso}"/>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_head_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.8" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="torso_left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_arm}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI}" upper="0" effort="30" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_arm*0.7}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Right Arm (mirror of left) -->
  <joint name="torso_right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="2"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_arm}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI}" upper="0" effort="30" velocity="2"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_arm*0.7}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="torso_left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_leg"/>
    <origin xyz="0.07 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.5"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_leg*0.6}"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI}" upper="0" effort="80" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_leg*0.4}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.06" ixy="0.0" ixz="0.0" iyy="0.06" iyz="0.0" izz="0.006"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="50" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_foot}"/>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.015"/>
    </inertial>
  </link>

  <!-- Right Leg (mirror of left) -->
  <joint name="torso_right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.07 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.5"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_leg*0.6}"/>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI}" upper="0" effort="80" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_leg*0.4}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.06" ixy="0.0" ixz="0.0" iyy="0.06" iyz="0.0" izz="0.006"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="50" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_foot}"/>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.015"/>
    </inertial>
  </link>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros2_control.so">
      <robot_namespace>/humanoid</robot_namespace>
      <robot_param>robot_description</robot_param>
    </plugin>
  </gazebo>

  <!-- IMU Sensor -->
  <gazebo reference="torso">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Camera Sensor -->
  <gazebo reference="head">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>head_camera_optical_frame</frame_name>
        <hack_baseline>0.07</hack_baseline>
        <distortion_k1>0.0</distortion_k1>
        <distortion_k2>0.0</distortion_k2>
        <distortion_k3>0.0</distortion_k3>
        <distortion_t1>0.0</distortion_t1>
        <distortion_t2>0.0</distortion_t2>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### 4.2 Isaac Sim Implementation

```python
# isaac_sim_humanoid.py
import omni
from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysicsSchemaTools
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

class IsaacSimHumanoid:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.setup_scene()

    def setup_scene(self):
        """Setup the Isaac Sim scene with humanoid robot"""
        # Add ground plane
        self.world.scene.add_ground_plane()

        # Add lighting
        self.setup_lighting()

        # Add humanoid robot
        self.add_humanoid_robot()

        # Set camera view
        set_camera_view(eye=[2.5, 2.5, 2.5], target=[0, 0, 1.0])

    def setup_lighting(self):
        """Setup lighting for the scene"""
        # Add dome light
        dome_light = UsdGeom.DomeLight.Define(self.world.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(500.0)
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    def add_humanoid_robot(self):
        """Add humanoid robot to the scene"""
        # For this example, we'll use a simple representation
        # In practice, you'd load a detailed humanoid model
        self.robot = Robot(
            prim_path="/World/Humanoid",
            name="humanoid_robot",
            usd_path=get_assets_root_path() + "/Isaac/Robots/Franka/franka_instanceable.usd"
        )
        self.world.scene.add(self.robot)

    def run_simulation(self):
        """Run the simulation with humanoid control"""
        self.world.reset()

        while True:
            self.world.step(render=True)

            # Get robot state
            if self.world.current_time_step_index % 100 == 0:
                joint_positions = self.robot.get_joint_positions()
                joint_velocities = self.robot.get_joint_velocities()

                # Apply control commands
                self.apply_control_commands(joint_positions, joint_velocities)

    def apply_control_commands(self, joint_positions, joint_velocities):
        """Apply control commands to the robot"""
        # Calculate desired joint positions based on walking pattern
        target_positions = self.calculate_walking_pattern()

        # Apply position control
        self.robot.set_joint_positions(target_positions)

    def calculate_walking_pattern(self):
        """Calculate walking pattern for humanoid"""
        # Simplified walking pattern - in practice this would be more complex
        time_step = self.world.current_time_step_index
        phase = (time_step % 200) / 200.0  # Normalized phase (0 to 1)

        # Generate walking pattern based on phase
        target_positions = np.zeros(9)  # Example for 9 DOF robot

        # Hip joints for walking
        target_positions[0] = 0.2 * np.sin(2 * np.pi * phase)  # Left hip
        target_positions[3] = 0.2 * np.sin(2 * np.pi * phase + np.pi)  # Right hip

        # Knee joints for walking
        target_positions[1] = 0.3 * np.sin(4 * np.pi * phase)  # Left knee
        target_positions[4] = 0.3 * np.sin(4 * np.pi * phase + np.pi)  # Right knee

        return target_positions

# Unity Robotics implementation would go here
# For Unity, we'd use the Unity Robotics Hub and ML-Agents

def main():
    """Main function to run Isaac Sim humanoid simulation"""
    humanoid_sim = IsaacSimHumanoid()
    humanoid_sim.run_simulation()

if __name__ == "__main__":
    main()
```

### 4.3 Gazebo Launch Configuration

```xml
<!-- launch/humanoid_gazebo.launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = LaunchConfiguration('world', default='empty.sdf')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open('/path/to/humanoid_model.urdf.xacro').read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from `/gazebo_ros/worlds`'
        ),
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity
    ])
```

## 5. Simulation → Real World Mapping

### 5.1 Physics Fidelity Considerations

**Mass Properties**: Ensure simulated masses, centers of mass, and moments of inertia match real hardware. Small discrepancies can lead to significant differences in dynamics.

**Friction Models**: Real surfaces have complex friction behaviors that may not be captured by simple Coulomb friction models in simulation.

**Actuator Dynamics**: Simulated motors should include realistic torque-speed curves, response times, and thermal limitations.

### 5.2 Sensor Accuracy

**Camera Calibration**: Simulated cameras should match real camera intrinsics, extrinsics, and noise characteristics.

**IMU Simulation**: Include bias, drift, and noise patterns that match real IMU sensors.

**Force Sensors**: Simulate the compliance and filtering characteristics of real force/torque sensors.

### 5.3 Domain Randomization

To bridge the sim-to-real gap, domain randomization techniques can be used:

```python
# domain_randomization.py
import numpy as np

class DomainRandomization:
    def __init__(self):
        self.param_ranges = {
            'mass_multiplier': (0.8, 1.2),
            'friction_coefficient': (0.3, 0.9),
            'com_offset': (-0.05, 0.05),
            'sensor_noise': (0.0, 0.01),
            'actuator_delay': (0.0, 0.05)
        }

    def randomize_parameters(self):
        """Randomize physics parameters for domain randomization"""
        randomized_params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            randomized_params[param] = np.random.uniform(min_val, max_val)
        return randomized_params

    def apply_randomization(self, simulation_env):
        """Apply randomized parameters to simulation"""
        params = self.randomize_parameters()

        # Apply mass randomization
        for link in simulation_env.links:
            original_mass = link.mass
            link.mass = original_mass * params['mass_multiplier']

        # Apply friction randomization
        for contact in simulation_env.contacts:
            contact.friction = params['friction_coefficient']

        # Apply sensor noise randomization
        simulation_env.imu.noise_std = params['sensor_noise']

        return simulation_env
```

## 6. Common Mistakes & Mental Models

### 6.1 Common Mistakes
- **Overfitting to simulation**: Training policies that work only in the specific simulation environment
- **Ignoring physics accuracy**: Using simplified physics that don't represent real-world behavior
- **Sensor model mismatches**: Using ideal sensors in simulation when real sensors have significant noise/latency
- **Computational complexity**: Creating simulations too complex to run in real-time
- **Inadequate validation**: Not testing simulation results on physical hardware

### 6.2 Mental Models for Success
- **Physics-first thinking**: Start with accurate physics modeling, then add complexity
- **Gradual complexity**: Begin with simple models and incrementally add realism
- **Validation loop**: Continuously validate simulation results against real-world data
- **Performance awareness**: Balance simulation fidelity with computational requirements

## 7. Mini Case Study: NVIDIA Isaac Sim in Humanoid Development

### 7.1 NVIDIA's Approach

NVIDIA's Isaac Sim platform demonstrates advanced simulation capabilities for humanoid robots:

**Photorealistic Rendering**: Using the Omniverse platform for high-fidelity graphics that enable training of computer vision systems directly in simulation.

**AI Training Integration**: Built-in reinforcement learning environments that can train complex humanoid behaviors.

**USD Format**: Universal Scene Description enables complex scene composition and asset sharing.

### 7.2 Real-World Application

Isaac Sim has been used to develop walking controllers for humanoid robots by:
- Training locomotion policies in diverse virtual environments
- Validating control algorithms before hardware deployment
- Generating synthetic training data for perception systems
- Testing failure scenarios safely in simulation

### 7.3 Lessons Learned

The success of advanced simulation platforms shows that:
- **High-fidelity rendering** enables effective computer vision training
- **Physics accuracy** is crucial for control system development
- **Domain randomization** helps bridge the sim-to-real gap
- **Integrated tooling** accelerates the development cycle

These insights have made simulation an essential component of modern humanoid robot development, enabling rapid iteration and safer development practices.