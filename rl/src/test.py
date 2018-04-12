#!/usr/bin/env python
# Test script

import environment
import gazeboInterface as gazebo
import rospy

if __name__ == '__main__':

	env = environment.Environment()

	rospy.init_node('test', anonymous=True)

	initState = env._reset()
	print('initState: {}'.format(initState))

	sampleState = env._sample()
	print('sampleState: {}'.format(sampleState))

	nextState, reward, isTerminal, _ = env._step(sampleState)
	print('nextState: {}, reward: {}'.format(nextState, reward))

	