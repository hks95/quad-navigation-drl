#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty

class GazeboInterface():
    
    def __init__(self):
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_proxy2 = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    
    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")
        
    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")
        
    def resetSim(self):
        rospy.wait_for_service('/gazebo/reset_world')
        # rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # self.reset_proxy()
            self.reset_proxy2()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")

