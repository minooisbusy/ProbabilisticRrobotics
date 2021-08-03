import enum
from numpy.core.numeric import zeros_like
import sys
import numpy as np
import math

import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
# Global variables
dt = 0.1 # time intervals
N = 200 # number of states
Q = np.diag([0.1,0.1,0.1]) **2
R = np.diag([0.1, 0.1, 0.1])**2

# Noise parameters
INPUT_NOISE = np.diag([10, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
def calc_input():
	v = 1.0
	yaw_rate = 0.1
	u = np.array([[v, yaw_rate]]).T
	return u

def observation(xTrue, xd, u, m): # This method is not a observation model h(x)
	xTrue = motion_model(xTrue, u)
	zs = find_map_around_robot(xTrue, m, 3) # True position based observation
	ud = u + GPS_NOISE@np.random.randn(2,1) # Noisy input!
	zds = zs + 0.1*np.random.randn(*zs.shape)
	xDR = motion_model(xd, ud)
	return xTrue, zs, zds, xDR, ud

def motion_model(xTrue, u):
	r = u[0]/u[1]
	g = np.array(
		[
			xTrue[0]-r*math.sin(xTrue[2]) + r * math.sin(xTrue[2] + u[1]*dt),
			xTrue[1]+r*math.cos(xTrue[2]) - r * math.cos(xTrue[2] + u[1]*dt),
			xTrue[2]+u[1]*dt
		]
	)
	return g


def observation_model(xPred, map):
	for m in map:
		delta = np.array([
			[m[0]-xPred[0]],m[1]-xPred[1],
			[math.atan2(delta[1],delta[0])-xPred[2]],
			])
		q = delta.T@delta
	
def make_map(r, N): # 원형 궤도 상에서 어떻게 하면 random 분포를 만들 수 있을까?
	# 벡터 표기로 방향을 랜덤으로 하고, 그 길이를 r로 제한하자
	theta = np.pi*2*np.random.randn(1)
	x = r*math.cos(theta) + np.random.randn()
	y = r*math.sin(theta) + r + np.random.randn()
	m = np.array([x, y])

	for _ in range(N-1):
		theta = np.pi*np.random.randn(1)
		x = r*math.cos(theta) + np.random.randn()
		y = r*math.sin(theta) +r + np.random.randn()
		m = np.vstack([m,np.array([x,y])])
	return m
def find_map_around_robot(xTrue, m, radius):
	zs = None
	for landmark in m:
		if (xTrue[0]-landmark[0])**2+(xTrue[1]-landmark[1])**2<radius**2:
			if zs is None:
				zs = landmark + 0.001*np.random.randn(2)
			else:
				zs = np.vstack([zs, landmark + 0.001*np.random.randn(2)])
	
	return zs

def ekf_estimation(xEst, PEst, zs, u, m):
	# Prediction
	xPred = motion_model(xEst,u)
	jG = jacob_g(xEst, u)
	PPred = jG@PEst@jG.T + Q

	# Update 
	#	find all fandkars k in the map m
	q = 0
	hat_z = np.zeros((3, len(m)))
	Hs = np.zeros((len(m),3,3))
	Psi = np.zeros((len(m),3,3))
	for k, landmark in enumerate(m):
		deltaX = landmark[0] - xPred[0]
		deltaY = landmark[1] - xPred[1]
		q = deltaX**2 + deltaY**2
		hat_z[0, k] = math.sqrt(q)
		hat_z[1, k] = math.atan2(deltaY, deltaX)-xPred[2]
		hat_z[2, k] = 0
		sqrtq = math.sqrt(q)
		Hs[k,:3,:3] = np.matrix([
			[sqrtq*deltaX, - sqrtq*deltaY, 0],
			[deltaY, deltaX, -1],
			[0, 0, 0]
		],dtype=float)/q
		Psi[k,:,:] = Hs[k,:,:]@PPred@Hs[k,:,:].T +Q
	
	Ks = np.zeros((len(zs),3,3))
	mapper = []
	for i,z in enumerate(zs):
		distance = np.zeros(len(m))
		for k in range(len(m)):
			invPsi = np.linalg.inv(Psi[k,:,:])
			dx = z-hat_z[:,k]
			distance[k] = dx@invPsi@dx.T
		j = np.argmin(distance)
		Ks[i,:,:] = PPred@Hs[j,:,:]@invPsi
		mapper.append(j)
	residual = np.zeros((3))
	for i, z in enumerate(zs):
		temp = Ks[i,:,:]@(z - hat_z[:,mapper[i]])
		residual += temp

	KH = np.zeros((3,3))
	for i, K in enumerate(Ks):
		KH += K@Hs[mapper[i],:,:]
	xEst = xPred + residual.reshape((3,1))
	PEst = (np.eye(3)-KH)@PPred
	return xEst, PEst


def jacob_g(x, u):
	G = np.matrix(
		[
			[1.0, 0.0, u[0]/u[1]*math.cos(x[2]) - u[0]/u[1]*math.cos(x[2]+u[1]*dt)],
			[0.0, 1.0, u[0]/u[1]*math.sin(x[2]) - u[0]/u[1]*math.sin(x[2]+u[1]*dt)],
			[0.0,0.0,1.0]
		], dtype=float
	)
	return G

def convert_z2feature(x, zs):
	output = np.zeros((len(zs), 3), dtype=float)
	for i, z in enumerate(zs):
		deltaX = z[0] - x[0]
		deltaY = z[1] - x[1]
		q = deltaX**2+deltaY**2
		output[i, 0] = math.sqrt(q)
		output[i, 1] = math.atan2(deltaY, deltaX)-x[2]
		output[i, 2] = 0
	return output



SIM_TIME = 30
show_animation = True
def main():
	# Initialize variables
	# states
	xEst = np.zeros((3,1)) # estimated state mean
	PEst = np.eye(3) # Estimated state covariance

	xTrue = np.zeros((3,1)) # true state
	xDR = np.zeros((3,1)) # Dead Reckoning: initial state evolution with noisy input
	time = 0.0
	# Generate states (True, Deadreckoning, Map)

	hxTrue = xTrue
	hxDR = xDR
	hxEst = xEst
	m=make_map(10, 100)
	hz = None
	
	while SIM_TIME >= time:
		time += dt
		u = calc_input() # linear and angular velocity
		xTrue, zs, zds, xDR, ud = observation(xTrue,xDR, u, m)

		zdfs = convert_z2feature(xTrue, zds)
		xEst, PEst = ekf_estimation(xEst, PEst, zdfs, ud, m)


		### Just plotting code
		hxTrue = np.hstack((hxTrue,xTrue))
		hxDR = np.hstack((hxDR, xDR))
		hxEst = np.hstack((hxEst, xEst))

		if show_animation:
			plt.cla()
			plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if (event.key == 'escape' or event.key == 'q') else None])
			# True pose trajectory
			plt.plot(hxTrue[0, :].flatten(),
					 hxTrue[1, :].flatten(), "-b")
			# Dead Reckoning position trajectory
			plt.plot(hxDR[0, :].flatten(),
					 hxDR[1, :].flatten(), "-k")
			plt.plot(hxEst[0, :].flatten(),
					 hxEst[1, :].flatten(), "-r")
			# Map
			plt.plot(m[:, 0],
					 m[:, 1],".g")
			# Real landmark position
			if len(zs) != 0:
				plt.plot(zs[:, 0].flatten(),
						zs[:, 1].flatten(), "oy")
				plt.plot(zds[:, 0].flatten(),
						zds[:, 1].flatten(), "xk")
			plt.axis("equal")
			plt.xlim(-20,20)
			plt.ylim(-5,25)
			plt.grid(True)
			plt.title('{},{},{}'.format(xEst[0],xEst[1],xEst[2]))
			plt.pause(0.001)
			
	return 0


if __name__=='__main__':
	main()