import scipy as sp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

a=1000.0# grids in domain
dz=1/a # discretization step
Modes=5.0# desired number of modes
BD = 1/dz*sp.sparse.spdiags(sp.vstack((-sp.ones(a), sp.ones(a))), sp.array([0, 1]), int(a), int(a)) # backward difference matrix
FD = -sp.sparse.dia_matrix.transpose(BD) #forward difference matrix through backward
Emat = -BD*FD #*Ex
Expect_k = 2*sp.pi # wave vector target to find resonance wavelength
k_sqvare, V = ssl.eigs(Emat, k=Modes, M=None, sigma=Expect_k**2)
k = sp.sort(sp.sqrt(k_sqvare)) # resonance wave vector
wavelength = 2*sp.pi/sp.real(k)
Q = sp.real(k)/(2*sp.imag(k)) # quality factor

fig = plt.figure()
plt.hold(True)
step_btw_plots=0.01+Modes
plt.plot(sp.arange(1,step_btw_plots,1), wavelength, '-',label="Numeriacal")
plt.plot(sp.arange(1,step_btw_plots,1), 4/(2*sp.arange(0,step_btw_plots-1,1)+1), 'o',label="Theory")
plt.legend(bbox_to_anchor=(0.7, 1.0), loc=2, ncol=1)
plt.ylabel('Wavelength')
plt.xlabel('Number of mode')
f, ax = plt.subplots(Modes,1, sharex=True, sharey=True)
for n in range(Modes):
    ax[n].plot(V[:,n],'-')
plt.ylabel('E_x')
plt.xlabel('Spatial')
plt.show()