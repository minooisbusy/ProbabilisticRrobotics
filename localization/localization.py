# Probabilsitic Rorotics Chapter 7. Localization algorithm(Table 7.2 and 7.3) implementation
# Original Author: Atsushi Sakai
# Author: Minwoo shin
# Date: 09/ Aug/ 21

import enum
from numpy.core.fromnumeric import squeeze
from numpy.core.numeric import zeros_like
import sys
import numpy as np

import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation as Rot

# Global variables
dt = 0.1 # time intervals
N = 200 # number of states

# Noise parameters
V_NOISE = 1.2
W_NOISE =np.deg2rad(30.0)
R_NOISE = 0.01 # observation noise
PHI_NOISE = np.deg2rad(30.0)
R = np.diag([V_NOISE, V_NOISE, W_NOISE])**2  	  # motion model uncertainty diag(x, y, theta)
Q = np.diag([30, 30, 1e16])**2		  			  # observation model uncertainty diag(radian, phi, signature)
INPUT_NOISE = np.diag([V_NOISE, W_NOISE]) ** 2

# Known correspondences switch, True = Known correspondences
KNOWN =  False
# Observation noise switch, True = measurements are noisy
NOISE = True 

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

	t = np.arange(0, 2 * np.pi + 0.1, 0.1)
	a = np.sqrt(abs(eigval[bigind]))
	b = np.sqrt(abs(eigval[smallind]))
	x = [a * np.cos(it) for it in t]
	y = [b * np.sin(it) for it in t]
	angle = np.arctan2(eigvec[1, bigind], eigvec[0, bigind])
	rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
	fx = rot @ (np.array([x, y]))
	px = np.array(fx[0, :] + xEst[0, 0]).flatten()
	py = np.array(fx[1, :] + xEst[1, 0]).flatten()
#	plt.plot(px, py, "--r",label=label)
	return px, py

def calc_input():
	v = 1.0
	yaw_rate = 0.1
	u = np.array([[v, yaw_rate]]).T # 2x1
	return u

def observation(xTrue, xd, u, m, SWITCH): # This method is not a observation model h(x)
	xTrue = motion_model(xTrue, u)
	zs = find_map_around_robot(xTrue, m, 3) # True position based observation
	ud = u + INPUT_NOISE@np.random.randn(2,1) # Noisy input!
	if SWITCH is True:
		z_noise =R_NOISE*np.insert(np.random.randn(*(zs.shape))[:,:2],2,0,axis=1)
		zds = zs + z_noise
	else:
		zds = zs
	
	xDR = motion_model(xd, ud)
	return xTrue, zs, zds, xDR, ud

def motion_model(xTrue, u):
	r = u[0]/u[1]
	g = np.array(
		[
			#xTrue[0]+u[0]*np.np.cos(xTrue[2])*dt,
			#xTrue[1]+u[0]*np.np.sin(xTrue[2])*dt,
			xTrue[0]-r*np.sin(xTrue[2])+r*np.sin(xTrue[2]+u[1]*dt),
			xTrue[1]+r*np.cos(xTrue[2])-r*np.cos(xTrue[2]+u[1]*dt),
			xTrue[2]+u[1]*dt
		]
	)
	return g

def jacob_g(x, u):
	r = u[0]/u[1] # r=v/{\omega}
	#G1=np.array([1.0, 0.0, -u[0]*dt*np.sin(x[2])],dtype=object)
	#G2=np.array([0.0, 1.0,  u[0]*dt*np.cos(x[2])],dtype=object)
	phi = x[2]
	w = u[1]
	r = np.squeeze(r)[()]
	phi = np.squeeze(phi)[()]
	w = np.squeeze(w)[()]

	G1=np.array([1.0, 0.0, -r*np.cos(phi)+r*np.cos(phi+w*dt)])
	G2=np.array([0.0, 1.0, -r*np.sin(phi)+r*np.sin(phi+w*dt)])
	G3=np.array([0.0,0.0,1.0])
	G = np.vstack([G1,G2,G3])
	return G


def jacob_h(x, m):
	deltaX = m[0] - x[0]
	deltaY = m[1] - x[1]
	q = deltaX**2 + deltaY**2
	deltaX = squeeze(deltaX)[()]
	deltaY = squeeze(deltaY)[()]
	q = squeeze(q)[()]
	sqrtq = np.sqrt(q)
	H1 = np.array([-deltaX/sqrtq, -deltaY/sqrtq, 0])
	H2 = np.array([deltaY/q, -deltaX/q, -1.0])
	H3 = np.array([0, 0, 0])
	H = np.vstack([H1,H2,H3])
	
	return H
	
def make_map(r, N): 
	noisy_r = r + np.random.randn()
	
	theta = np.pi*2*np.random.randn()
	x = noisy_r*np.cos(theta) 
	y = noisy_r*np.sin(theta) + r 
	m = np.array([x, y, 0])

	for i in range(1,N):
		theta = np.pi*np.random.randn()
		noisy_r = r + np.random.randn()
		x = noisy_r*np.cos(theta) 
		y = noisy_r*np.sin(theta) +r
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
	xPred, PPred = motion_update(xEst,u, PEst) # Prediction with noisy control input $u$
	
	mapper = [] # Initialize Correspondence variable $c_t^k$
	xEst, PEst, mapper = measurement_update(xPred, PPred, zs[:,:], m, mapper)

	return xEst, PEst, mapper




def convert_z2feature(x, zs):
	output = np.zeros((len(zs), 3))
	for i, z in enumerate(zs):
		deltaX = z[0] - x[0]
		deltaY = z[1] - x[1]
		q = deltaX**2+deltaY**2
		output[i, 0] = np.sqrt(q)
		output[i, 1] = np.arctan2(deltaY, deltaX)-x[2]
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

		# $delta$
		deltaX = landmark[0] - x[0]
		deltaY = landmark[1] - x[1]

		# $q$
		q = deltaX**2 + deltaY**2 

		# squared root of $q$
		sqrtq = np.sqrt(q)

		# \hat{z}
		hat_z[0, k] = sqrtq
		hat_z[1, k] = np.arctan2(deltaY, deltaX)-x[2]
		hat_z[2, k] = landmark[2]

		# $H$
		Hs[k,:3,:3]=jacob_h(x, landmark)

		# $\Psi$
		Psi[k,:3,:3] =(Hs[k,:3,:3]@PPred@Hs[k,:3,:3].T +Q).astype(float)
	
	xEst = x
	PEst = PPred.copy()
	for i, z in enumerate(zs):
		# $\argmin dx^T*\Psi_i^{-1}*dx
		invPsi = np.linalg.inv(Psi[i,:,:].astype(float))
		j = int(match_features(z, hat_z, invPsi, known=KNOWN))
		mapper.append(j)

		# Last step
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
		distance[k] = (dx@invPsi@dx.T)/(np.linalg.det(2*np.pi*invPsi)) # exponential and squared root are monotonically increase, so i peel of these guys.
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
	
	fig, axes = plt.subplots(2,2)
	while SIM_TIME >= time:
		time += dt
		u = calc_input() # linear and angular velocity
		xTrue, zs, zds, xDR, ud = observation(xTrue,xDR, u, m, SWITCH=NOISE)

		zdfs = convert_z2feature(xTrue, zds) # position to feature
		xEst, PEst, mapper = ekf_estimation(xEst, PEst, zdfs, ud, m, zs)


		### belows are Just plotting code
		hxTrue = np.hstack((hxTrue,xTrue))
		hxDR = np.hstack((hxDR, xDR))
		hxEst = np.hstack((hxEst, xEst))
		hPxcoord = np.hstack((hPxcoord, PEst[0,0]))
		hPycoord = np.hstack((hPycoord, PEst[1,1]))

		
		if show_animation:
			
			for i in range(2):
				for j in range(2):
					axes[i][j].cla()
			#fig.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if (event.key == 'escape' or event.key == 'q') else None])
			# Map
			axes[0][0].plot(m[:, 0],
					 m[:, 1],".k")
			# Observed Real landmark position
			if len(zs) != 0:
#				plt.plot(zs[:, 0].flatten(),
#						zs[:, 1].flatten(), "+g")
			# Observed Noisy landmakr position
				axes[0][0].plot(zds[:, 0].flatten(),
						zds[:, 1].flatten(), "xy", label="Noisy observations")
			# True pose trajectory
			axes[0][0].plot(hxTrue[0, :].flatten(),
					 hxTrue[1, :].flatten(), "--",color="blue", label="True trajectory")
			axes[0][0].plot(hxTrue[0,-1],
					 hxTrue[1,-1], ".b", label="Current true position")
			# Dead Reckoning position trajectory
			axes[0][0].plot(hxDR[0, :].flatten(),
					 hxDR[1, :].flatten(), "--k", label="Dead Reckoning")
			axes[0][0].plot(hxDR[0,-1],
					 hxDR[1,-1], ".r", label="Current DR position")

			# Estimated pose trajectory
			axes[0][0].plot(hxEst[0, :].flatten(),
					 hxEst[1, :].flatten(), "-", color="lime",label="Estimated")
			axes[0][0].plot(hxEst[0,-1],
					 hxEst[1,-1], ".",color="lime", label="Current Est position")


			axes[0][0].set_aspect("equal")
			axes[0][0].set_xlim(-20,20)
			axes[0][0].set_ylim(-5,25)
			axes[0][0].grid(True)
			axes[0][0].set_title('Simulation plot')
			px, py = plot_covariance_ellipse(xEst,PEst)
			axes[0][0].plot(px, py, "--r",label='State covariance')
			# correspondences, red: outlier
			for i, z in enumerate(zds[:,:]):
				j = mapper[i]
				if  j < 0:
					j = abs(j)
					axes[0][0].plot([z[0],m[j][0]],
						[z[1], m[j][1]], "-r")
				else:
					axes[0][0].plot([z[0],m[j][0]],
						[z[1], m[j][1]], "-g")
			#axes[0][0].legend()
			axes[0][1].set_title('x-coord variance')
			#plt.ylim(0,0.3)
			axes[0][1].plot(hPxcoord,color="k")
			axes[0][1].grid(True)
			axes[1][0].set_title('y-coord variance')
			axes[1][0].plot(hPycoord,color="k")
			axes[1][0].grid(True)
			#plt.title('velocity variance, {}'.format(xEst[3]))
			#plt.plot(hPvel,color="k")
			#plt.grid(True)
			#plt.title('matrix color')
			axes[1][1].matshow(np.abs(PEst), cmap=plt.cm.Blues)
			for (i, j), z in np.ndenumerate(PEst):
				axes[1][1].text(j, i, '{0:.1f}'.format(z,ha='center',va='center'))
			plt.pause(0.00001)
			key = None
			while key =='c':
				key = plt.waitforbuttonpress(0)
			if key == 'escape':
				sys.exit()
		
	key = plt.waitforbuttonpress(0)


			
	return 0


if __name__=='__main__':
	main()
