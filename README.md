# quad-navigation-drl

This repository is meant to help anyone interested in applying deep reinforcement learning techniques for quadcopter navigation. In our work we train an agent to navigate to a given goal location while minimizing the energy it consumes. This is achieved by adding a battery constraint to the agent. More details can be learned from this report: https://drive.google.com/open?id=1MFA5vo_2P6awaqf-R4eUkYQ6VAy-cJRi.

We simulate a quadcopter in the hector_gazebo simulator developed by TU-Darmstadt. We have developed the interface to perform most functions similar to an OpenAI Gym environment. We currently implement the Deep Deterministic Policy Gradient (DDPG) algorithm to train the agent. 

0. Install(from source) the hector-gazebo simulator from: 
http://wiki.ros.org/hector_quadrotor/Tutorials/Quadrotor%20outdoor%20flight%20demo

Use this command to install other dependencies if you don't have them: 
`rosdep install --from-paths src --ignore-src -r -y`

1. For using pr2 teleop:
`git clone https://github.com/PR2/pr2_apps.git`

2. For using teleop twist keyboard:

`sudo apt-get install ros-indigo-teleop-twist-keyboard`

3. To launch the basic demo:

`roslaunch hector_quadrotor_demo outdoor_flight_gazebo.launch`

4. For teleop use either of the following (the second one is more complicated but is needed to get off the ground)
`roslaunch pr2_teleop teleop_keyboard.launch`
`rosrun teleop_twist_keyboard teleop_twist_keyboard.py`

5. Execute the python publisher:
`chmod +x pub.py` (only the first time in the same folder as pub.py)
`python pub.py`

If running the above script in a tensorflow virtual environment, might need to do:
`pip install rospkg catkin_pkg`

HACK: Change the outdoor_flight_gazebo.launch file to include an argument as follows:
`<arg name="z" value="5.0"/>`
