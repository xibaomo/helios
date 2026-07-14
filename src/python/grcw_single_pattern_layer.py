'''
Transmission and reflection of single pattern layer
'''
import grcwa 
import numpy as np 

nG = 23**2
L = 400.
L1 = [L,0.]
L2 = [0.,L]
freq = 1/193 
theta = 10./180.*np.pi 
phi = 0. 

Qabs = np.inf 
freqcmp = freq*(1+1j/2/Qabs)

Nx = 400
Ny = 400 

#eps for patterned layer 
a=200
epgrid = np.ones((Nx,Ny),dtype = float) 
s = int(L/2-a/2)
# epgrid[s:s+a,s:int(s+a/2)] = 2.612**2
yy, xx = np.meshgrid(np.arange(int(L)), np.arange(int(L)), indexing='ij')
center = L/2
radius = a/2
mask = (xx-center)**2 + (yy-center)**2 <= radius**2
epgrid = np.ones((int(L), int(L)), dtype=complex)
epgrid[mask] = 2.612**2

obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=1)
obj.Add_LayerUniform(1,1.563**2)
obj.Add_LayerGrid(56,Nx,Ny)
obj.Add_LayerUniform(1,1.)
obj.Init_Setup(Gmethod=1) 

planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order=0)
obj.GridLayer_geteps(epgrid.flatten())

R,T = obj.RT_Solve(normalize=1)
print('R=',R,', T=',T,', R+T=',R+T)


