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
		self.goalPos = [-10.0, 10.0, 3.0]
		self.goal_threshold = 0.5
		self.crash_reward = -5
		self.goal_reward = 50 #reduce to 20 #allow more aggressive motions?

		self.num_states = 3
		self.num_actions = 3

		self.max_altitude = 10.0
		self.min_altitude = 0.5

		self.max_x =  15.0
		self.min_x = -15.0

		self.max_y =  15.0
		self.min_y = -15.0

		self.debug = debug

		self.prev_state = []

		self.battery = 200
		self.battery_exp = 1.1
		# self.plotState = np.zeros((self.num_states,))

		# imu_sub = message_filters.Subscriber("/raw_imu", Imu)
		# pose_sub = message_filters.Subscriber("/ground_truth_to_tf/pose", PoseStamped)
		# vel_sub = message_filters.Subscriber("/fix_velocity", Vector3Stamped)
		
		# fs = [imu_sub, pose_sub, vel_sub]
		# queue_size = 2  # The both have a frquency of 100Hz, we won't need more than 2 messages of each
		# slop = 0.07  # (in sec) The above two topics are by observation at max 0.05 out of sync
		# ats = message_filters.ApproximateTimeSynchronizer(fs, queue_size, slop)  #, allow_headerless=False)

		# ats.registerCallback(self.sensor_callback)

	def getRandomGoal(self):

		# Completely random goal might be too tough
		# random_theta = random.uniform(0, np.pi)
		# radius = 5
		# z_min = 1.5
		# z_max = 3.5

		# goal_x = radius * np.cos(random_theta)
		# goal_y = radius * np.sin(random_theta)
		# goal_z = random.uniform(z_min, z_max)

		# return [goal_x, goal_y, goal_z]

		goalPos1 = [0.0,  -5.0, 2.0]
		goalPos2 = [0.0,  5.0, 2.0]
		goalPos3 = [5.0,  0.0, 2.0]
		goalPos4 = [-5.0, 0.0, 2.0]

		goalPos = [goalPos1, goalPos2, goalPos3, goalPos4]

		random_choice = random.randint(0,3)

		return goalPos[random_choice]

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

		###########
		#  ROHIT  #
		###########
		nextState = [pose_.position.x, pose_.position.y, pose_.position.z]

		# print('next state : {} {} {} {} {} {}'.format(nextState[0],nextState[1],nextState[2],nextState[3],nextState[4],nextState[5],nextState[6])) 
		self.plotState = np.vstack((self.plotState, np.asarray(nextState)[0:3]))

		self.prev_state = nextState

		return nextState, reward, isTerminal, []

	def _reset(self):

		# 1st: resets the simulation to initial values
		self.gazebo.resetSim()
		self.battery = 200
		# 2nd: Unpauses simulation
		self.gazebo.unpauseSim()
		# 
		# self.goalPos = self.getRandomGoal()
		# 3rd: Don't want to start the agent from the ground
		self.takeoff()

		# 4th: Get init state
		# TODO: Should initial state have some randomness?
		initStateData, _, _, _ = self.takeObservation()

		###########
		#  ROHIT  #
		###########
		initState = [initStateData.pose.pose.position.x, 
					 initStateData.pose.pose.position.y, 
					 initStateData.pose.pose.position.z]
		# print('init state : {} {} {} {} {} {}'.format(initState[0],initState[1],initState[2],initState[3],initState[4],initState[5],initState[6])) 
		
		self.plotState = np.asarray(initState)[0:3]
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
		currentPos = [poseData.position.x, poseData.position.y, poseData.position.z]
		
		if self.debug:
			print('distance from goal: {}'.format(error))
		# reward += -error

		if error < self.goal_threshold:
			reward += self.goal_reward
			reachedGoal = True
		
		else:
			# pdb.set_trace()
			# reward = reward + min(5/(error),50) #100 is clipping value
			reward = reward + (np.linalg.norm(np.subtract(self.prev_state[0:3], self.goalPos)) - np.linalg.norm(np.subtract(currentPos, self.goalPos)))
			# print ("dist reward  {} ".format((np.linalg.norm(np.subtract(self.prev_state, self.goalPos)) - np.linalg.norm(np.subtract(currentPos, self.goalPos)))))
			# print("self.battery_drain(velData) {} ".format(self.battery_drain(velData)))
			# reward = reward + self.battery_drain(velData)/100 #also try scaling just by 10
			# reward = 10
			reachedGoal = False
			# reward += -error			

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

	def battery_drain(self, vel):
		velocity = np.array([1+abs(vel.vector.x), 1+abs(vel.vector.y), 1+abs(vel.vector.z)])
		velocity = np.linalg.norm(velocity)
		return -(velocity)**self.battery_exp

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

		self.battery += self.battery_drain(velData)

		roll, pitch, yaw = self.quaternion_to_euler_angle(imuData.orientation.x, imuData.orientation.y, imuData.orientation.z, imuData.orientation.w)

		pitch_bad = not(-self.max_incl < pitch < self.max_incl)
		roll_bad = not(-self.max_incl < roll < self.max_incl)
		altitude_bad = poseData.position.z > self.max_altitude or poseData.position.z < self.min_altitude
		x_bad = poseData.position.x > self.max_x or poseData.position.x < self.min_x
		y_bad = poseData.position.y > self.max_y or poseData.position.y < self.min_y
		# print('motorData.on: {}'.format(motorData.on))  # MotorData message doesn't really work
		# print('------------------------------------')
		if altitude_bad or pitch_bad or roll_bad or x_bad or y_bad:
			# rospy.loginfo ("(Terminating Episode: Unstable quad) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+","+str(x_bad)+","+str(y_bad)+")")
			print('Unstable quad')
			done = True
			reward = self.crash_reward  # TODO: Scale this down?
		# elif self.battery <= 0:
		# 	print ('battery dead')
		# 	reward = self.crash_reward
		# 	done = True
		else:  # TODO: Should we get a reward if we terminate?
			reward, reachedGoal = self.getReward(poseData, imuData, velData)
			if reachedGoal:
				print('Reached Goal!')
				done = True

		if self.debug:
			print('Step Reward: {} battery level {}'.format(reward,self.battery))

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

