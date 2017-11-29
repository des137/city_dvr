# 1D harmonic-oszillator benchmark for the DVR
# comments: 
# (i)  converged(!) results for the SINE DVR are offset
# but the splitting between energy EVs is correct
# (ii) the choice of parameters has significant effect
# on the grid sisze which is necessary for convergence
import time
import numpy as np
import numpy.linalg
from sympy.physics.secondquant import KroneckerDelta as delt

_EPS = np.finfo(float).eps

np.set_printoptions(linewidth=300,suppress=True)


# parameters defining the physical system
#----------------------------------------
hbarc   = 197.327
mn      = 938.
# spherically symmetric oszillator strength
K       = 12.
#
# lattice set-up
#-----------------------------
# implemented DVR bases: ['SINE','SINEap']
dvrbasis = 'SINE'
#
dim = 3
Lr = 7
Nr = 7
# for all box bases with f(a)=f(b)=0:
# endpoints are not elements of the grid!
dv = (Nr-1)**3
print dv**2
exit()
#
dr = float(Lr)/float(Nr+1)
L0 = -float(Lr)/2.
#
POT  = np.zeros((dv,dv))
KIN  = np.zeros((dv,dv))
#
xx,yy,zz = np.meshgrid(np.arange(1,Nr),np.arange(1,Nr),np.arange(1,Nr))
#
s = 0
tic = time.time()
for ix in range(xx.shape[0]):
	for iy in range(yy.shape[0]):
		for iz in range(zz.shape[0]):
			#print '(%+d,%+d,%+d) --'%(ix,iy,iz),
			a = np.array([ix,iy,iz]) ; a+=1
			r = 0
			for ipx in range(xx.shape[0]):
				for ipy in range(yy.shape[0]):
					for ipz in range(zz.shape[0]):
						#print '(%+d,%+d,%+d)'%(ipx,ipy,ipz)
						b = np.array([ipx,ipy,ipz]) ; b+=1
						POT[r,s] = 0.0
						if np.array_equal(a,b):
							POT[r,s] = 0.5*K*((a[0]*dr+L0)**2+(a[1]*dr+L0)**2+(a[2]*dr+L0)**2)
						for i in range(dim):
							if a[i]==b[i]:
								if dvrbasis=='SINE':
									# E_kin ME in SINE-basis
									KIN[r,s] += np.pi**2/(2.*Lr**2)*( ((2.*Nr**2+1)/3.)-np.sin(np.pi*a[i]/float(Nr))**(-2))*delt(a[(i+1)%dim],b[(i+1)%dim])*delt(a[(i+2)%dim],b[(i+2)%dim])
								if dvrbasis=='SINEap':
									# E_kin ME in SINE-basis; assuming (-inf,inf) interval
									KIN[r,s] += (1./dr**2)*(-1)**(a[i]-b[i])*np.pi**2/3.*delt(a[(i+1)%dim],b[(i+1)%dim])*delt(a[(i+2)%dim],b[(i+2)%dim])
		 					else:
								if dvrbasis=='SINE':
									# E_kin ME in SINE-basis
									KIN[r,s] += (-1)**(a[i]-b[i])*np.pi**2/(2.*Lr**2)*(np.sin(np.pi*(a[i]-b[i])/(2.*Nr))**(-2)-np.sin(np.pi*(a[i]+b[i])/(2.*Nr))**(-2))*delt(a[(i+1)%dim],b[(i+1)%dim])*delt(a[(i+2)%dim],b[(i+2)%dim])
								if dvrbasis=='SINEap':
									# E_kin ME in SINE-basis; assuming (-inf,inf) interval
		 							KIN[r,s] += (1./dr**2)*2./(a[i]-b[i])**2*delt(a[(i+1)%dim],b[(i+1)%dim])*delt(a[(i+2)%dim],b[(i+2)%dim])
						r += 1
			s += 1
#
tac = time.time()
#
KIN *= hbarc**2/(2*mn)
#
HAM = (KIN + POT)
EV = np.sort(np.linalg.eigvals(HAM))
#
print 'DVR:'
print np.real(EV)[:6]

print np.real(np.diff(EV)[:6])
#
Eana = np.sort(np.array([[[ (nx+ny+nz+1.5)*hbarc*np.sqrt(K/mn) for nx in range(20) ] for ny in range(20) ]for nz in range(20)]).flatten())
print 'ANA:'
print Eana[:6]
print np.diff(Eana)[:6]
#print EV[:10]-Eana[:10]
toc = time.time()
print 'ME filling takes %12.2f time units,'%(tac-tic)
print 'Diagonalization  %12.2f time units'%(toc-tac)
exit()
#-----------------------------