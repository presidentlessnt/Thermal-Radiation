#! /usr/bin/python3
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d

## EXPERIMENTO 3 ##
RW=.6 #Ohm
E3=np.array([[1,2,3,4,5,6,7,8,9,10]
	,[.60,.78,.93,1.06,1.19,1.30,1.41,1.51,1.60,1.69]
	,[.1,.5,1.3,2.4,3.7,5.1,6.7,8.4,10.1,12]]).T # V,A, rad(mV)

def fun3(x):
    return -.0001*x**6+.0072*x**5-.1941*x**4+2.6315*x**3-20.566*x**2+269.28*x+50.653 

RvT=np.loadtxt('RvT-LSB.txt')
f=interp1d(RvT[:,0],RvT[:,1])

n3=len(E3)
nE3=np.zeros((n3,3))
for i in range(n3):
    nE3[i,2]=E3[i,0]/E3[i,1]/RW	#resistencia (Ohm)
    nE3[i,0]=np.log10(f(nE3[i,2]))	#temperatura (°C)
    nE3[i,1]=np.log10(E3[i,2])		#radiacion   (mV)

mE3=np.zeros((n3,3))
for i in range(n3):
    mE3[i,2]=E3[i,0]/E3[i,1]/RW	#resistencia (Ohm)
    mE3[i,0]=fun3(nE3[i,2])	#temperatura (°C)
    mE3[i,1]=E3[i,2]		#radiacion   (mV)


# Lineal function
def func(p,x):
    b,c = p
    return b*x+c


# Model object
quad_model = odr.Model(func)


# Create a RealData object
data = odr.RealData(nE3[:,0], nE3[:,1])


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

x_fit = nE3[:,0]
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw= func(popt_dw, x_fit)

#plot
'''
fig, ax = plt.subplots(1)
plot(nE3[:,0], nE3[:,1],'o', label='Data')
plot(x_fit, fit, 'r', lw=2, label='Ajuste lineal \n $m:%1.1f \pm %1.1f$'%(slope,pmslope))
ax.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='Intervalo 5-sigma')
legend(loc=4)
grid(ls='--',color='grey',lw=.5)
'''

figure(2)
plot(mE3[:,0], mE3[:,1],'o', label='Data')
plot(mE3[:,0],10**(popt[1])*mE3[:,0]**popt[0],'r',lw=2,label='Ajuste lineal \n $m:%1.1f \pm %1.1f$'%(slope,pmslope))
xlabel('Temperatura [K]')
ylabel('Radiación [mV]')
grid(ls='--',color='grey',lw=.5)
xscale('log')
yscale('log')
legend()

figure(3)
plot(mE3[:,0],mE3[:,1],'o',label='Data')
xlabel('Temperatura [K]')
ylabel('Radiación [mV]')
grid(ls='--',lw=.5,color='grey')
legend()

show()