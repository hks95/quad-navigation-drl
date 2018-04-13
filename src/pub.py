#!/usr/bin/env python
# license removed for brevity
import rospy
from hector_uav_msgs.msg import Altimeter
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Quaternion, Point, Pose, Vector3, Vector3Stamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, Range, Image
from hector_uav_msgs.msg import Altimeter
import message_filters
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import gazeboInterface as gazebo

def imu_callback(data):
    
    global angle
    angle = Quaternion()
    angle.x = data.orientation.x
    angle.y = data.orientation.y
    angle.z = data.orientation.z
    angle.w = data.orientation.w

    return

def altimeter_callback(data):

    global quad_height

    quad_height = data.altitude

    return

def image_callback(data):

    global image
    image_height = data.height
    image_width = data.step

    image = np.fromstring(data.data, np.uint8)
    image = np.reshape(image, [image_height, image_width])
    # rospy.loginfo(image)

    return

def pose_callback(data):

    global pose

    pose = Pose()

    pose.position.x = data.pose.position.x
    pose.position.y = data.pose.position.y
    pose.position.z = data.pose.position.z

    pose.orientation.x = data.pose.orientation.x
    pose.orientation.y = data.pose.orientation.y
    pose.orientation.z = data.pose.orientation.z
    pose.orientation.w = data.pose.orientation.w

    return

def range_callback(data):

    global range_

    range_ = data.range

    return

def sensor_callback(imu_data, pose_data):
# Common callback for multiple sensors when using Approx Time Sync 

    global pose, angle

    angle = Quaternion()

    angle.x = imu_data.orientation.x
    angle.y = imu_data.orientation.y
    angle.z = imu_data.orientation.z
    angle.w = imu_data.orientation.w

    pose = Pose()

    pose.position.x = pose_data.pose.position.x
    pose.position.y = pose_data.pose.position.y
    pose.position.z = pose_data.pose.position.z

    pose.orientation.x = pose_data.pose.orientation.x
    pose.orientation.y = pose_data.pose.orientation.y
    pose.orientation.z = pose_data.pose.orientation.z
    pose.orientation.w = pose_data.pose.orientation.w

    return

def control():
    # Function to publish velocity commands to the quad

    rate = rospy.Rate(10)
    count = 0
    msg = Twist()

    while not rospy.is_shutdown():
        # TODO: Put this in an init function
        if(count < 10):
            msg.linear.z = 0.5
            rospy.loginfo('Lift off')

        # TODO: Put this in a shutdown function
        elif(count > 100):
            rospy.loginfo('Stop')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            break

        else:
            rospy.loginfo('Move forward')
            msg.linear.x = 0.5
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            print('angle.x: %.2f' % angle.x)
            print('pose.position.x: %.2f' % pose.position.x)

        pub_.publish(msg)
        count = count + 1
        rate.sleep()

    return

def listener():

    rospy.init_node('pub', anonymous=True)

    # Bunch of sensor measurements, will think of what to use later
    # rospy.Subscriber("/raw_imu", Imu, imu_callback)
    # rospy.Subscriber("/altimeter", Altimeter, altimeter_callback)
    # rospy.Subscriber("/front_cam/camera/image", Image, image_callback)
    # rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, pose_callback)
    # rospy.Subscriber("/sonar_height", Range, range_callback)

    # Considering Approzimate Time Synchronization
    # http://docs.ros.org/api/message_filters/html/python/#message_filters.TimeSynchronizer

    imu_sub = message_filters.Subscriber("/raw_imu", Imu)
    pose_sub = message_filters .Subscriber("/ground_truth_to_tf/pose", PoseStamped)
    
    fs = [imu_sub, pose_sub]
    queue_size = 2  # The both have a frquency of 100Hz, we won't need more than 2 messages of each
    slop = 0.07  # (in sec) The above two topics are by observation at max 0.05 out of sync
    ats = message_filters.ApproximateTimeSynchronizer(fs, queue_size, slop)  #, allow_headerless=False)

    ats.registerCallback(sensor_callback)
    
    global pub_
    pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    
    control()
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()  # No real point in starting the agent without any sensor data
    except rospy.ROSInterruptException:
        pass