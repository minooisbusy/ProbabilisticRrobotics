import numpy as np
import math

import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
# Global variables
dt = 0.1 # time intervals
N = 200 # number of states
Q = np.diag([1,1,1])
R = Q.copy()
def calc_input():
	v = 1.0
	yaw_rate = 0.1
	u = np.array([[v, yaw_rate]]).T
	return u

def observation(xTrue, xd, u):
	xTrue = motion_model(xTrue, u)

def motion_model(xTrue, u):
	g = np.array


def motion_model(xTrue, u):
	g = np.array


def main():
	# Initialize variables
	# states
	xEst = np.zeros((3,1)) # estimated state mean
	PEst = np.eye(4) # Estimated state covariance

	xTrue = np.zeros((3,1)) # true state
	xDR = np.zeros((3,1)) # Dead Reckoning: initial state evolution with noisy input
	time = 0.0
	# Generate states (True, Deadreckoning, Map)
	for n in N:
		time += dt
		u = calc_input()

		xTrue, z, xDR, ud = observation(xTrue, xDR, u)
		

	return 0


if __name__=='__main__':
	main()