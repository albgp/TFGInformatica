import scipy
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

fu = lambda alpha: lambda u:  scipy.exp(2*scipy.pi*1j*u*alpha)


plotFunction = lambda alpha: np.absolute(complex_quadrature(fu(alpha),0,1)[0])
plotFunctionComplex = lambda alpha: complex_quadrature(fu(alpha),0,1)[0]

print(plotFunction(1./2))
a = np.arange(-0.5, 0.5, 0.001)
fval=np.array([plotFunction(t) for t in a])
plt.plot(a,fval)
plt.xlabel(r'$\frac{\{rz\}}{r}$',
          fontsize=27)
plt.ylabel(r'$f(\frac{\{rz\}}{r})=\Big|\int_0^1e^{-2\pi i \frac{\{rz\}}{r} u}\, du\Big|$', fontsize=27)
#plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
#plt.title(r"\TeX\ is Number "
#          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#          fontsize=16, color='gray')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
plt.savefig('../Diagramas/integralTocha.eps', format='eps', dpi=1000)
plt.show()

#fig,ax = plt.subplots()
#vals=[plotFunctionComplex(t) for t in a]
#ax.scatter([val.real for val in vals ],[val.imag for val in vals])
#plt.show()
