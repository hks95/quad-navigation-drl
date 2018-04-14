from __future__ import absolute_import
import rospy
# from hector_uav_msgs.msg import Altimeter
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Quaternion, Point, Pose, Vector3, Vector3Stamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, Range, Image
from hector_uav_msgs.msg import Altimeter, MotorStatus
from nav_msgs.msg import Odometry
import message_filters
import matplotlib.pyplot as plt
import numpy as np
import gazeboInterface as gazebo
import time
import random
import math
import pdb

class Environment():

	def __init__(self, debug):

		self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.gazebo = gazebo.GazeboInterface()
		self.running_step = 0.05  # Convert to ros param
		self.max_incl = np.pi/2
		
		self.vel_min = -2.0
		self.vel_max = 2.0
		self.goalPos = [0.0, -5.0, 2.0]
		self.goal_threshold = 0.5
		self.crash_reward = -20
		self.goal_reward = 20

		self.num_states = 3
		self.num_actions = 3

		self.max_altitude = 5.0
		self.min_altitude = 0.5

		self.max_x =  10.0
		self.min_x = -10.0

		self.max_y =  10.0
		self.min_y = -10.0

		self.debug = debug

		self.prev_state = []

		# self.plotState = np.zeros((self.num_states,))

		# imu_sub = message_filters.Subscriber("/raw_imu", Imu)
		# pose_sub = message_filters.Subscriber("/ground_truth_to_tf/pose", PoseStamped)
		# vel_sub = message_filters.Subscriber("/fix_velocity", Vector3Stamped)
		
		# fs = [imu_sub, pose_sub, vel_sub]
		# queue_size = 2  # The both have a frquency of 100Hz, we won't need more than 2 messages of each
		# slop = 0.07  # (in sec) The above two topics are by observation at max 0.05 out of sync
		# ats = message_filters.ApproximateTimeSynchronizer(fs, queue_size, slop)  #, allow_headerless=False)

		# ats.registerCallback(self.sensor_callback)

	def _step(self, action):

		# Input: action
		# Output: nextState, reward, isTerminal, [] (not sending any debug information)

		vel = Twist()
		vel.linear.x = action[0]
		vel.linear.y = action[1]
		vel.linear.z = action[2]

		if self.debug:
			print('vel_x: {}, vel_y: {}, vel_z: {}'.format(vel.linear.x, vel.linear.y, vel.linear.z))
		
		self.gazebo.unpauseSim()
		self.pub.publish(vel)
		time.sleep(self.running_step)
		poseData, imuData, velData, motorData = self.takeObservation()
		self.gazebo.pauseSim()

		pose_ = poseData.pose.pose
		reward, isTerminal = self.processData(pose_, imuData, velData, motorData)
		nextState = [pose_.position.x, pose_.position.y, pose_.position.z]
		self.plotState = np.vstack((self.plotState, np.asarray(nextState)))

		self.prev_state = nextState

		return nextState, reward, isTerminal, []

	def _reset(self):

			# 1st: resets the simulation to initial values
			self.gazebo.resetSim()

			# 2nd: Unpauses simulation
			self.gazebo.unpauseSim()

			# 3rd: Don't want to start the agent from the ground
			self.takeoff()

			# 4th: Get init state
			# TODO: Should initial state have some randomness?
			initStateData, _, _, _ = self.takeObservation()

			initState = [initStateData.pose.pose.position.x, initStateData.pose.pose.position.y, initStateData.pose.pose.position.z]
			self.plotState = np.asarray(initState)
			self.prev_state = initState
			# 5th: pauses simulation
			self.gazebo.pauseSim()

			return initState

	def _sample(self):

		vel_x = random.uniform(self.vel_min, self.vel_max)
		vel_y = random.uniform(self.vel_min, self.vel_max)
		vel_z = random.uniform(self.vel_min, self.vel_max)

		return [vel_x, vel_y, vel_z]

	def takeObservation(self):
		# TODO: Using wait_for_message for now, might change to ApproxTimeSync later 

		poseData = None
		while poseData is None:
		  try:
			  # poseData = rospy.wait_for_message('/ground_truth_to_tf/pose', PoseStamped, timeout=5)
			  poseData = rospy.wait_for_message('/ground_truth/state', Odometry, timeout=5)
		  except:
			  rospy.loginfo("Current drone pose not ready yet, retrying to get robot pose")

		velData = None
		while velData is None:
		  try:
			  velData = rospy.wait_for_message('/fix_velocity', Vector3Stamped, timeout=5)
		  except:
			  rospy.loginfo("Current drone velocity not ready yet, retrying to get robot velocity")

		imuData = None
		while imuData is None:
		  try:
			  imuData = rospy.wait_for_message('/raw_imu', Imu, timeout=5)
		  except:
			  rospy.loginfo("Current drone imu not ready yet, retrying to get robot imu")

		motorData = None
		# while motorData is None:
		#   try:
		#       motorData = rospy.wait_for_message('/motor_status', MotorStatus, timeout=5)
		#   except:
		#       rospy.loginfo("Current drone motor status not ready yet, retrying to get robot motor status")
		
		return poseData, imuData, velData, motorData

	def _distance(self, pose):

		currentPos = [pose.position.x, pose.position.y, pose.position.z]
		if self.debug:
			print('currentPos: {}'.format(currentPos))
		
		# dist = np.linalg.norm(np.subtract(currentPos, self.goalPos))
		err = np.subtract(currentPos, self.goalPos)
		w = np.array([1, 1, 4])
		err = np.multiply(w,err)
		dist = np.linalg.norm(err)
		return dist
	
	def getReward(self, poseData, imuData, velData):
		# Input: poseData, imuData
		# Output: reward according to the defined reward function

	# TODO: Change the error to weight the z_error higher

		reward = 0

		error = self._distance(poseData)
		
		if self.debug:
			print('distance from goal: {}'.format(error))
		# reward += -error

		if error < self.goal_threshold:
			reward += self.goal_reward
			reachedGoal = True
		
		else:
			# pdb.set_trace()
			reward = reward + min(1/(error),10) #100 is clipping value
			# reward = 10
			reachedGoal = False
			# reward += -error
			
			# Add other rewards here
			


	# TODO: Probably need to make a 3D equivalent of this
		# angletoGoal = np.arctan2(np.abs(poseData.position.y - self.goalPos[1]), np.abs(poseData.position.x - self.goalPos[2]))
		# currentAngle = np.arctan2(velData.vector.y, velData.vector.x)

		# # if self.debug:
		#     # print('arctan2({},{}), arctan2({},{})'.format(np.abs(poseData.position.y - self.goalPos[1]), np.abs(poseData.position.x - self.goalPos[2]), velData.vector.y, velData.vector.x))
		#     # print('angletoGoal: {}, currentAngle: {}'.format(angletoGoal, currentAngle))

		# if(angletoGoal - np.pi/6 < currentAngle < angletoGoal + np.pi/6):
		#     reward += 1
		# else:
		#     reward -= 5

		return reward, reachedGoal

	def quaternion_to_euler_angle(self, x, y, z, w):
		ysqr = y * y
		
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + ysqr)
		X = math.atan2(t0, t1)
		
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		Y = math.asin(t2)
		
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (ysqr + z * z)
		Z = math.atan2(t3, t4)
		
		return X, Y, Z

	def processData(self, poseData, imuData, velData, motorData):

		done = False
		
		# euler = tf.transformations.euler_from_quaternion([imuData.orientation.x, imuData.orientation.y, imuData.orientation.z, imuData.orientation.w])
		# roll = euler[0]
		# pitch = euler[1]
		# yaw = euler[2]

		roll, pitch, yaw = self.quaternion_to_euler_angle(imuData.orientation.x, imuData.orientation.y, imuData.orientation.z, imuData.orientation.w)

		pitch_bad = not(-self.max_incl < pitch < self.max_incl)
		roll_bad = not(-self.max_incl < roll < self.max_incl)
		altitude_bad = poseData.position.z > self.max_altitude or poseData.position.z < self.min_altitude
		x_bad = poseData.position.x > self.max_x or poseData.position.x < self.min_x
		y_bad = poseData.position.y > self.max_y or poseData.position.y < self.min_y
		# print('motorData.on: {}'.format(motorData.on))  # MotorData message doesn't really work
		# print('------------------------------------')
		if altitude_bad or pitch_bad or roll_bad or x_bad or y_bad:
			rospy.loginfo ("(Terminating Episode: Unstable quad) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+","+str(x_bad)+","+str(y_bad)+")")
			done = True
			reward = self.crash_reward  # TODO: Scale this down?

		else:  # TODO: Should we get a reward if we terminate?
			reward, reachedGoal = self.getReward(poseData, imuData, velData)
			if reachedGoal:
				print('Reached Goal!')
				done = True

		if self.debug:
			print('Step Reward: {}'.format(reward))

		return reward,done

	def takeoff(self):

		# rate = rospy.Rate(10)
		count = 0
		msg = Twist()

		# while not rospy.is_shutdown():
		while count < 2:
			msg.linear.z = 0.5
			# rospy.loginfo('Lift off')

			self.pub.publish(msg)
			count = count + 1
			time.sleep(1.0)

		msg.linear.z = 0.0
		self.pub.publish(msg)
		if self.debug:
			print('Take-off sequence completed')
		return


