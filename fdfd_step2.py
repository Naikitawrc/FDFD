import scipy as sp
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

eps_slabs=[13.0,1.0]#dielectric contrast
a=1000.0# grids in domain
Modes=5# desired number of modes
dz=1/a # discretization step
k_slab = 2*sp.pi*sp.arange(0.01,0.501,0.005) # k-vector contrast between layers with eps1 and eps2
FD = 1/dz*sp.sparse.spdiags(sp.vstack((-sp.ones((1,a)), sp.ones((1,a)))), sp.array([0, 1]), int(a), int(a)) #forward difference matrix
FD = sp.sparse.lil_matrix(FD)
eps=sp.ones(a)
eps[0:int(a/2)] = 1/eps_slabs[0]
eps[a/2:int(a)] = 1/eps_slabs[1]
eps = sp.sparse.spdiags(eps, 0, a, a) # diagonal matrix with 1/eps
k = sp.zeros((Modes, k_slab.size), dtype=complex)
for ik in range(k_slab.size):
    k0=k_slab[ik]
    FD[int(a)-1,0] = sp.cos(k0*1)/dz # creating pereodic boundary conditions
    BD = -sp.sparse.lil_matrix.transpose(FD) # backward difference matrix
    Expect_k = k0/sp.sqrt((eps_slabs[0]+eps_slabs[1])/2) # ecpectation k
    k_sqvare, V = ssl.eigs(-eps*BD*FD, k=Modes, M=None, sigma=Expect_k**2)
    k[:,ik] = sp.sqrt(k_sqvare)

wavelength = 2*sp.pi/sp.real(k)
Q = sp.real(k)/(2*sp.imag(k))
fig = plt.figure()
for n in range(Modes):
    plt.hold(True)
    plt.plot(k_slab/(2*sp.pi), k[n,:]/(2*sp.pi) ,'-')
plt.ylabel('k/(2pi)')
plt.xlabel('Wave vector, ka/(2pi)')
f, ax = plt.subplots(Modes,1, sharex=True, sharey=True)

for n in range(Modes):
    ax[n].plot(V[:,n],'-')
plt.ylabel('E_x')
plt.xlabel('Spatial')
plt.show()