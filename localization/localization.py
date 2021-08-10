# Probabilsitic Rorotics Chapter 7. Localization algorithm(Table 7.1 and 7.2) implementation
# Original Author: Atsushi Sakai
# Author: Minwoo shin
# Date: 09/ Aug/ 21

import enum
from numpy.core.numeric import zeros_like
import sys
import numpy as np
import math

import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation as Rot

# Global variables
dt = 0.1 # time intervals
N = 200 # number of states
R = np.diag([1, 1, 0])**2
Q = np.diag([0.1, 0.1, np.deg2rad(30.0)])**2

# Noise parameters
INPUT_NOISE = np.diag([1, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.1, 0.1]) ** 2

# Known correspondences
KNOWN = True 
def plot_covariance_ellipse(xEst, PEst):
	Pxy = PEst[0:2, 0:2]
	Pxy=np.array(Pxy, dtype=float)
	eigval, eigvec = np.linalg.eig(Pxy)

	if eigval[0] >= eigval[1]:
	    bigind = 0
	    smallind = 1
	else:
	    bigind = 1
	    smallind = 0

	t = np.arange(0, 2 * math.pi + 0.1, 0.1)
	a = math.sqrt(abs(eigval[bigind]))
	b = math.sqrt(abs(eigval[smallind]))
	x = [a * math.cos(it) for it in t]
	y = [b * math.sin(it) for it in t]
	angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
	rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
	fx = rot @ (np.array([x, y]))
	px = np.array(3*fx[0, :] + xEst[0, 0]).flatten()
	py = np.array(3*fx[1, :] + xEst[1, 0]).flatten()
	plt.plot(px, py, "--r")

def calc_input():
	v = 1.0
	yaw_rate = 0.1
	u = np.array([[v, yaw_rate]]).T # 2x1
	return u

def observation(xTrue, xd, u, m, max_measure): # This method is not a observation model h(x)
	xTrue = motion_model(xTrue, u)
	zs = find_map_around_robot(xTrue, m, 3) # True position based observation
	zs = zs[:max_measure,:]
	ud = u + INPUT_NOISE@np.random.randn(2,1) # Noisy input!
	z_noise =0.01*np.insert(np.random.randn(*(zs.shape))[:,:2],2,0,axis=1)
	zds = zs #+ z_noise
	
	xDR = motion_model(xd, ud)
	return xTrue, zs, zds, xDR, ud

def motion_model(xTrue, u):
	r = u[0]/u[1]
	g = np.array(
		[
			#xTrue[0]+u[0]*np.math.cos(xTrue[2])*dt,
			#xTrue[1]+u[0]*np.math.sin(xTrue[2])*dt,
			xTrue[0]-r*math.sin(xTrue[2])+r*math.sin(xTrue[2]+u[1]*dt),
			xTrue[1]+r*math.cos(xTrue[2])-r*math.cos(xTrue[2]+u[1]*dt),
			xTrue[2]+u[1]*dt
		]
	)
	return g

def jacob_g(x, u):
	r = u[0]/u[1] # r=v/{\omega}
	#G1=np.array([1.0, 0.0, -u[0]*dt*math.sin(x[2])],dtype=object)
	#G2=np.array([0.0, 1.0,  u[0]*dt*math.cos(x[2])],dtype=object)
	G1=np.array([1.0, 0.0, -r*math.cos(x[2])+r*math.cos(x[2]+u[1]*dt)],dtype=object)
	G2=np.array([0.0, 1.0, -r*math.sin(x[2])+r*math.sin(x[2]+u[1]*dt)],dtype=object)
	G3=np.array([0.0,0.0,1.0],dtype=object)
	G = np.vstack([G1,G2,G3])
	return G


def jacob_h(x, m):
	deltaX = m[0] - x[0]
	deltaY = m[1] - x[1]
	q = deltaX**2 + deltaY**2
	sqrtq = math.sqrt(q)
	H1 = np.array([-deltaX/sqrtq, -deltaY/sqrtq, 0],dtype=object)
	H2 = np.array([deltaY/q, -deltaX/q, -1.0],dtype=object)
	H3 = np.array([0, 0, 0],dtype=object)
	H = np.vstack([H1,H2,H3])
	
	return H
	
def make_map(r, N): 
	noisy_r = r + np.random.randn()
	
	theta = np.pi*2*np.random.randn(1)
	x = noisy_r*math.cos(theta) 
	y = noisy_r*math.sin(theta) + r 
	m = np.array([x, y, 0])

	for i in range(1,N):
		theta = np.pi*np.random.randn(1)
		noisy_r = r + np.random.randn()
		x = noisy_r*math.cos(theta) 
		y = noisy_r*math.sin(theta) +r
		id = i
		m = np.vstack([m,np.array([x,y, i])])
	return m
def find_map_around_robot(xTrue, m, radius):
	zs = None
	for landmark in m:
		if (xTrue[0]-landmark[0])**2+(xTrue[1]-landmark[1])**2<radius**2:
			landmark 
			if zs is None:
				zs = landmark 
			else:
				zs = np.vstack([zs, landmark ])
	
	return zs

def ekf_estimation(xEst, PEst, zs, u, m, testZ):
	# Prediction Line 2 ~ 4
	# Line 2
	xPred, PPred = motion_update(xEst,u, PEst) # Prediction with noisy control input $u$
	
	mapper = [] # Initialize Correspondence variable $c_t^k$
	xEst, PEst, mapper = measurement_update(xPred, PPred, zs[:,:], m, mapper)

	return xEst, PEst, mapper




def convert_z2feature(x, zs):
	output = np.zeros((len(zs), 3), dtype=object)
	for i, z in enumerate(zs):
		deltaX = z[0] - x[0]
		deltaY = z[1] - x[1]
		q = deltaX**2+deltaY**2
		output[i, 0] = math.sqrt(q)
		output[i, 1] = math.atan2(deltaY, deltaX)-x[2]
		output[i, 2] = z[2]
	return output

def motion_update(x, u, P):
	xPred = motion_model(x,u)
	G = jacob_g(x, u)
	PPred = G@P@G.T + R

	return xPred, PPred

def measurement_update(x, PPred, zs, m, mapper):
	N=len(m)
	hat_z = np.zeros((3, N))
	Hs = np.zeros((N,3,3))
	Psi = np.zeros((N,3,3))
	for k, landmark in enumerate(m):
		# Line 7
		deltaX = landmark[0] - x[0]
		deltaY = landmark[1] - x[1]
		# Line 8
		q = deltaX**2 + deltaY**2 
		sqrtq = math.sqrt(q)
		# Line 9
		hat_z[0, k] = sqrtq
		hat_z[1, k] = math.atan2(deltaY, deltaX)-x[2]
		hat_z[2, k] = landmark[2]
		# Line 10
		Hs[k,:3,:3]=jacob_h(x, landmark)
		# Line 11
		Psi[k,:3,:3] =(Hs[k,:3,:3]@PPred@Hs[k,:3,:3].T +Q).astype(float)
	
	xEst = x
	PEst = PPred.copy()
	for i, z in enumerate(zs):
		invPsi = np.linalg.inv(Psi[i,:,:].astype(float))
		j = int(match_features(z, hat_z, invPsi, known=KNOWN))
		mapper.append(j)
		invPsi = np.linalg.inv(Psi[j,:,:].astype(float))
		K = (PEst@Hs[j,:,:].T@invPsi).astype(float)
		residual = (z-hat_z[:,j]).reshape(3,1)
		xEst += K@residual.astype(float)
		PEst = ((np.eye(len(xEst))-K@Hs[j,:,:]))@PEst


	return xEst, PEst, mapper

def match_features(z, hat_z, invPsi, known=False):
	N = len(hat_z[0,:]) # number of Landmark (size of map)
	distance = np.zeros(N)
	for k in range(N): # 1:N search which can cause multi-mapping
		dx = z-hat_z[:,k]
		distance[k] = dx@invPsi@dx.T/np.linalg.det(invPsi)
	j = np.argmin(distance)
	if (z[2] != hat_z[2,j]) and KNOWN:
		return np.argwhere(hat_z[2,:]==z[2])
	return j



SIM_TIME = 60
show_animation = True
def main():
	# Initialize variables
	# states
	xEst = np.zeros((3,1)) # estimated state mean
	PEst = 0.003*np.eye(3) # Estimated state covariance

	xTrue = np.zeros((3,1)) # true state
	xDR = np.zeros((3,1)) # Dead Reckoning: initial state evolution with noisy input
	time = 0.0
	# Generate states (True, Deadreckoning, Map)

	hxTrue = xTrue
	hxDR = xDR
	hxEst = xEst
	hxPred = xEst
	hPxcoord = PEst[0,0]
	hPycoord = PEst[1,1]
	m=make_map(10, 150)
	hz = None
	
	while SIM_TIME >= time:
		time += dt
		u = calc_input() # linear and angular velocity
		xTrue, zs, zds, xDR, ud = observation(xTrue,xDR, u, m,99)

		zdfs = convert_z2feature(xTrue, zds) # position to feature
		xEst, PEst, mapper = ekf_estimation(xEst, PEst, zdfs, ud, m, zs)


		### Just plotting code
		hxTrue = np.hstack((hxTrue,xTrue))
		hxDR = np.hstack((hxDR, xDR))
		hxEst = np.hstack((hxEst, xEst))
		hPxcoord = np.hstack((hPxcoord, PEst[0,0]))
		hPycoord = np.hstack((hPycoord, PEst[1,1]))

		

		if show_animation:
			plt.subplot(1,3,1)
			plt.cla()
			plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if (event.key == 'escape' or event.key == 'q') else None])
			# True pose trajectory
			plt.plot(hxTrue[0, :].flatten(),
					 hxTrue[1, :].flatten(), "-b")
			# Dead Reckoning position trajectory
			plt.plot(hxDR[0, :].flatten(),
					 hxDR[1, :].flatten(), "--k")
			plt.plot(hxEst[0, :].flatten(),
					 hxEst[1, :].flatten(), "--c")

			# Map
			plt.plot(m[:, 0],
					 m[:, 1],".g")
			# Real landmark position
			if len(zs) != 0:
				plt.plot(zs[:, 0].flatten(),
						zs[:, 1].flatten(), "+y")
				plt.plot(zds[:, 0].flatten(),
						zds[:, 1].flatten(), "xk")
			for i, z in enumerate(zds[:,:]):
				plt.plot([z[0],m[mapper[i]][0]],
					  [z[1], m[mapper[i]][1]], "-r")
			plt.axis("equal")
			plt.xlim(-20,20)
			plt.ylim(-5,25)
			plt.grid(True)
			plt.title('{},{},{}'.format(xEst[0],xEst[1],xEst[2]))
			plot_covariance_ellipse(xEst,PEst)
			plt.subplot(1,3,2)
			plt.ylim(0,0.3)
			plt.plot(hPxcoord,color="k")
			plt.subplot(1,3,3)
			plt.ylim(0,0.3)
			plt.plot(hPycoord,color="k")
			plt.pause(0.001)
			key = None
			while key =='c':
				key = plt.waitforbuttonpress(0)
			if key == 'escape':
				sys.exit()


			
	return 0


if __name__=='__main__':
	main()
