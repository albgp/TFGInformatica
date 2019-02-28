import numpy as np
import matplotlib.pyplot as plt
from odeintw import odeintw # This library is downloaded from https://github.com/WarrenWeckesser/odeintw
from scipy.linalg import hadamard
import numba as nb
import time

# Wrapper for timing
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{}  {:10.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed

#**** Constants ****#
regularizationConstant=10**-3
stepsConstant=100000


#**** Code ****#

@nb.njit(fastmath=True)
def norm(l):
    s = 0.
    for i in range(l.shape[0]):
        s += np.abs(l[i])**2
    return np.sqrt(s)

"""

"""
@timeit
def factor(N, T=100000): # Integer to factor
    assert N>0 #Must be a positive one
    print("Factoring N={}".format(N))
    n=int(np.log2(N))+1 #Bits needed to represent N in binary
    dim=2*n #Dimensions for the complex Hilbert space in which the evolution will take place

    print("Dimensions for the Hilbert space: 2^{}={}".format(dim,2**dim))

    v0=np.full(2**dim, 1/2**(dim//2)).astype(np.complex128) #Initial ground state, which is a uniform superposition H^(2^dim)|0>

    #Let's create the hadamard 2^dim-dimensional gate with the scipy library
    Had=hadamard(2**dim, dtype=complex)/2**dim


    reg=regularizationConstant

    # The function computes the value for the initial hamiltonian H_0 over a input vector v
    def H0_mult(v):
        h=lambda n: 0 if n==0 else 1 #Let's define the h function just as done in the project report
        ret=np.full(2**dim, 0.+0.j) #We start with a vector of all zeros
        for i in range(2**dim): # and we iterate as showed in the equation 4.4.2 <z|H^(2dim)|v>|z>
            vi=np.full(2**dim, 0.+0.j) 
            vi[i]=1.+0.j #We create |z>
            Hvi=np.dot(Had,vi) # Then we calculate |H^(2dim)|v>
            ret+=h(i)*np.vdot(Hvi,v)*Hvi # And least we add <z|H^(2dim)|v>|z>
        return ret

    # The function computes the value for the final hamiltonian H_f over a input vector v
    def Hf_mult(v):
        Q= lambda x,y: N**2*(N-x*y)**2+x*(x-y)**2 # Integer function from the paper by Kieu
        
        ret=np.array(v)
        for i in range(2**(dim//2)):
            for j in range(2**(dim//2)): # We iterate over all pair (i,j)~2^dim*i+j in the range
                ind=2**(dim//2)*i+j
                ret[ind]*=Q(i,j)
        #return ret/np.linalg.norm(ret)
        return ret*reg

    #The hamiltonian we are looking for is the interpolation over time of them both.
    def H(v,t):
        ret = (1-t/T)*H0_mult(v)+t/T*Hf_mult(v)
        #If the result of applying the hamiltonian over the vector is too small, then we normalize it
        return ret*(-1j)/(1 if np.linalg.norm(ret)<0.0001 else np.linalg.norm(ret))

    # We keep the interpolated hamiltonian without regularization for testing.
    def Hnoreg(v,t):
        ret = (1-t/T)*H0_mult(v)+t/T*Hf_mult(v)/reg
        return ret*(-1j)

    # And finally, our computation begins

    steps=stepsConstant #How many steps will our integration consists of
    t=np.linspace(0,T,num=steps) #So we subdivide the inverval into *steps* pieces

    res=odeintw(H,v0,t) # And we integrate the differential multi-dimensional equation.
    #print(res[0])
    #print(np.linalg.norm(res[0]))
    pos=np.argmax(np.array([np.linalg.norm(ress) for ress in res[-1]]))

    x,y=pos//2**(dim//2), pos%2**(dim//2) #We calculate the divisors from the found eigenstate.

    #Auxiliary functions to print stuff.
    def plotEnergy(res,t ):
        energies=[np.linalg.norm(Hnoreg(res[i], t[i])) for i in range(len(res)) ]
        plt.plot(t,energies)
        plt.savefig("N={},T={}_energies.eps".format(N,T))
        plt.show()
        print(energies[-1])

    def plotThings(res,t):
        #my_xticks = ["({},{})".format(i//2**(dim//2),i%2**(dim//2) ) for i in range(2**dim)]
        #plt.xticks(range(2**dim), my_xticks)
        plt.plot(range(2**dim), [np.abs(x) for x in res],'bo')
        plt.savefig("N={},T={}.eps".format(N,T))
        plt.show()

    def plotNorm(res):
        norms=[np.linalg.norm(i) for i in res]
        #print(norms)
        plt.plot(t,norms)
        plt.show()

    plotEnergy(res, t)
    plotThings(res[-1],t)
    #plotNorm(res)


    return x,y #We return both divisors.

for T in [10000]:
    N=3
    x,y=factor(N,T=T)
    assert x*y==N #So we check that our algorithm calculated the correct divisors for N
    print("Worked! {}={}x{}".format(N,x,y))
    print()
