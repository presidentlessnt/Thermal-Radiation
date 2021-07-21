#! /usr/bin/python3
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d

def fun3(x):
    return -.0001*x**6+.0072*x**5-.1941*x**4+2.6315*x**3-20.566*x**2+269.28*x+50.653 

RW=.6 #Ohm

RvT=np.loadtxt('RvT-LSB.txt')
f=interp1d(RvT[:,0],RvT[:,1])


## EXPERIMENTO 2 ##
v2=10 #V
a2=11.69 #mA
E2aux=np.array([[10,20,30,40,50,60,70,80,90,100],[.0,.0,.1,.1,.0,.0,.0,.0,.0,.0]]).T # Temp ambiental
#E2=np.array([[1.5,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100]
#	    ,[42.2,31.3,25.8,22.4,19.3,16.7,15,11.1,8.9,7,5.8,4.9,4.1,3.7,3.3,2.5,2,1.15,.9,.6,0,.3,.2,.1,0,0,0,0,0]]).T
E2=np.array([[1.5,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,14,16,18,20,25,30,35,40,45,50]
	    ,[42.2,31.3,25.8,22.4,19.3,16.7,15,11.1,8.9,7,5.8,4.9,4.1,3.7,3.3,2.5,2,1.15,.9,.6,.5,.3,.2,.1]]).T


n2=len(E2)
nE2=np.zeros((n2,2))
for i in range(n2):
    nE2[i,0]=np.log10(E2[i,0])	#distancia (cm)
    nE2[i,1]=np.log10(E2[i,1])	#radiacion (mV)

# Lineal function
def func(p,x):
    b,c = p
    return b*x+c


# Model object
quad_model = odr.Model(func)


# Create a RealData object
data = odr.RealData(nE2[:,0], nE2[:,1])


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

slope=popt[0]
pmslope=perr[0]

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = nE2[:,0]
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw= func(popt_dw, x_fit)


#plot
fig, ax = plt.subplots(1)
plot(nE2[:,0], nE2[:,1],'o', label='Data')
plot(x_fit, fit, 'r', lw=2, label='Ajuste lineal \n $m:%1.3f \pm %1.3f$'%(slope,pmslope))
xlabel('$log_{10}$ Distancia [cm]')
ylabel('$log_{10}$ Radiación [mV]')
grid(ls='--',color='grey',lw=.5)
legend(loc=1)


figure(2)
plot(E2[:,0], E2[:,1],'o', label='Data')
plot(E2[:,0],10**(popt[1])*E2[:,0]**popt[0],'r',lw=2,label='Ajuste lineal \n $m:%1.3f \pm %1.3f$'%(slope,pmslope))
#plot(E2[:,0],10**(popt[1])*E2[:,0]**-2,color='C2',lw=2,label='Teórico')#='Ajuste lineal \n $m:%1.3f \pm %1.3f$'%(slope,pmslope))
xlabel('Distancia [cm]')
ylabel('Radiación [mV]')
grid(ls='--',color='grey',lw=.5)
legend(loc=1)

show()