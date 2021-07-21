#! /usr/bin/python3
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d


RvT=np.loadtxt('RvT-leslie.txt')
f=interp1d(RvT[:,0],RvT[:,1])

## EXPERIMENTO 1 ##
## PARTE 1 ##
Too=23+273 #K
E1=np.array([[35.5,21.5,17.3,13.5]
	,[65,70.28,85.65,96.75]
	,[3.9,6.1,10.7,14.5]
	,[3.8,5.4,10.3,13.7]
	,[0.5,0.9,1.6,1.9]
	,[0.1,0.5,0.9,0.8]]) # Resis kOhm, Temp °C, rad (mV) {negra,blanca,mate,pulido}

nE1=np.copy(E1)
for i in range(4):
    nE1[0,i]=f(E1[0,i]*1e3)

#nE1[0]=[49,61,67,74]

mE1=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        mE1[i,j]=E1[i+2,j]/E1[2,j]*100

print(nE1,'\n')
print(mE1)

## PARTE 2 ##
# Otras radiaciones
E1b=np.array([[.05,.04,.03,.05,.3,1.1]])
#mano,medio ambiente,regla metal,compa,negra+v.traslucido,negra+v.polarizado

'''
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
'''

figure(3)
plot((nE1[0]+273)**4-(Too)**4,E1[2],'o',label='negra')
plot((nE1[0]+273)**4-(Too)**4,E1[3],'o',label='blanca')
plot((nE1[0]+273)**4-(Too)**4,E1[4],'o',label='mate')
plot((nE1[0]+273)**4-(Too)**4,E1[5],'o',label='pulida')
grid(ls='--',lw=.5,color='grey')
xlabel('Temperatura [K]')
ylabel('Radiación [mV]')
#xlim(3e9,8e9)
ylim(0,16)
legend(title='Superficie')


figure(4)
plot((nE1[0]+273),E1[2],'o',label='negra')
plot((nE1[0]+273),E1[3],'o',label='blanca')
plot((nE1[0]+273),E1[4],'o',label='mate')
plot((nE1[0]+273),E1[5],'o',label='pulida')
grid(ls='--',lw=.5,color='grey')
xlabel('Temperatura [K]')
ylabel('Radiación [mV]')
#xlim(3e9,8e9)
ylim(0,16)
legend(title='Superficie')


figure(5)
plot((nE1[0]),E1[2],'o',label='negra')
plot((nE1[0]),E1[3],'o',label='blanca')
plot((nE1[0]),E1[4],'o',label='mate')
plot((nE1[0]),E1[5],'o',label='pulida')
grid(ls='--',lw=.5,color='grey')
xlabel('Temperatura [°C]')
ylabel('Radiación [mV]')
xlim(45,75)
ylim(0,16)
legend(title='Superficie')


show()