#! /usr/bin/python3
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d


RvT=np.loadtxt('RvT-leslie.txt')
f=interp1d(RvT[:,0],RvT[:,1])

## EXPERIMENTO 4 ##
RcL=87.2 # Ohm
Too=23+273 # Kelvin
E4=np.array([[70.3,60.1,55.7,48.9,43,35.3,30.3,22.4,15.9,9.4,4.5]
	,[1.7,3.3,3.8,5,5.7,7.9,9,12,16.3,19.2,21.3]]).T # R_termistor(kOhm), rad(mV)



mE4=np.array([[32,36,38,41,44,49,52,60,69,84,107]
	,[1.7,3.3,3.8,5,5.7,7.9,9,12,16.3,19.2,21.3]]).T # T_termistor(°C), rad(mV)

n4=len(E4) 
nE4=np.zeros((n4,3))
for i in range(n4):
    nE4[i,0]=(f(E4[i,0]*1e3)+273)**4+Too**4
    nE4[i,1]=E4[i,1]
    nE4[i,2]=f(E4[i,0]*1e3)

print(nE4)

# Lineal function
def func(p,x):
    b,c = p
    return b*x+c


# Model object
quad_model = odr.Model(func)


# Create a RealData object
data = odr.RealData(nE4[:,0], nE4[:,1])


# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[1., 1.])


# Run the regression.
out = odr.run()


#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta
print("fit parameter 1-sigma error")
print("———————————–")
out.pprint()

for i in range(len(popt)):
    print(str(popt[i])+"+-"+str(perr[i]))  

slope=popt[0]*1e10
pmslope=perr[0]*1e10

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = nE4[:,0]
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw= func(popt_dw, x_fit)


#plot
fig, ax = plt.subplots(1)
plot(nE4[:,0], nE4[:,1],'o', label='Data')
plot(x_fit, fit, 'r', lw=2, label='Ajuste lineal \n $m:%1.1f \pm %1.1f$'%(slope,pmslope))
legend(loc=4)
xlabel('$T^4-T_{amb}^4$ [$K^4$]')
ylabel('Radiación [mV]')
grid(ls='--',color='grey',lw=.5)
xlim(1.6e10,3e10)
ylim(0,25)

show()