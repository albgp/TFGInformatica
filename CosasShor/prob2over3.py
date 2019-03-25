"""
This script plots the upper bound for the number of runs of the quantum order finding algortithm needed
to ensure that we get 'r' (using the mechanism of continous fractions) with a probability greater than 2/3.
"""

__author__="Aberto García"
__license__ = "GPL"
__email__ = "alberto.garcia12@um.es"

import fractions
import numpy as np
from matplotlib import pyplot as plt

trgt_file="../Diagramas/iteracionesNecesarias.eps"


"""
Inefficient Euler's totient
"""
def phi(n):
    amount = 0
    for k in range(1, n):
        if fractions.gcd(n, k) == 1:
            amount += 1
    return amount

"""
Given r, returns the minimum k such that
1-(1-(r/(3log(r)))^k>=2/3
"""
def getNumberOfNeededExecutions(r):
    phiVal=phi(r)
    prob=phiVal/(3*r)
    for i in range(1,1000):
        if 1-(1-prob)**i>=2./3:
            return i


rVals=np.array(range(5,10000))
neededExec=np.array([getNumberOfNeededExecutions(r) for r in rVals])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.scatter(rVals, neededExec)
plt.xlabel(r'Orden $r$', fontsize=27)
plt.ylabel(r'N\'{u}mero de iteraciones necesarias', fontsize=27)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
plt.savefig(trgt_file, format='eps', dpi=1000)
plt.show()
