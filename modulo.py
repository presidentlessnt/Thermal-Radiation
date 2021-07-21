#! /usr/bin/python3
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


####################### SEMANA 2 ############################
#	   SOLUCION ECUACIONES NO LINEALES
#############################################################

## METODO INCREMENTAL ##
def inc(fun,x0,dx):
    ite=0
    xr=x0
    A=np.empty((0,3),dtype=np.float64)
    while fun(xr)[0]*fun(xr+dx)[0]>0:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite
        AX[0,2]=xr
        AX[0,1]=abs((xr-(xr+dx))/xr)
        xr+=dx
        A=np.append(A,AX,axis=0)
        ite+=1
    return A


## METODO BISECTRICES ##
def bisec(fun,xl,xu,es):
    ite=0
    Err=1
    xr=xl
    A=np.empty((0,3),dtype=np.float64)
    while Err>es:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite+1
        AX[0,2]=xr
        if ite==0:
            AX[0,1]=1
        else:
            AX[0,1]=abs((AX[0,2]-A[ite-1,2])/AX[0,2])
        Err=AX[0,1]
        xr=(xl+xu)/2
        vl=fun(xl)[0]*fun(xr)[0]
        if vl<0:
            xu=xr
        else:
            xl=xr
        A=np.append(A,AX,axis=0)
        ite+=1
    return A

## METODO FALSA POSICION ##
def fake(fun,xl,xu,es):
    ite=0
    Err=1
    xr=xu
    A=np.empty((0,3),dtype=np.float64)
    while Err>es:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite+1
        AX[0,2]=xr
        if ite==0:
            AX[0,1]=1
        else:
            AX[0,1]=abs((AX[0,2]-A[ite-1,2])/AX[0,2])
        Err=AX[0,1]
        xr=xu-((fun(xu)[0]*(xl-xu))/(fun(xl)[0]-fun(xu)[0]))
        vl=fun(xl)[0]*fun(xr)[0]
        if vl<0:
            xu=xr
        else:
            xl=xr
        A=np.append(A,AX,axis=0)
        ite+=1
    return A

## METODO PUNTO FIJO ##
def fix(fun,x0,es):
    ite=0
    Err=1
    xr=x0
    A=np.empty((0,3),dtype=np.float64)
    while Err>es:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite+1
        AX[0,2]=xr
        if ite==0:
            AX[0,1]=1
        else:
            AX[0,1]=abs((AX[0,2]-A[ite-1,2])/AX[0,2])
        Err=AX[0,1]
        xr=-fun(x0)[0]+x0
        x0=xr
        A=np.append(A,AX,axis=0)
        ite+=1
    return A

## METODO NEWTON-RAPHSON ##
def nera(fun,x0,es):
    ite=0
    Err=1
    xr=x0
    A=np.empty((0,3),dtype=np.float64)
    while Err>es:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite+1
        AX[0,2]=xr
        if ite==0:
            AX[0,1]=1
        else:
            AX[0,1]=abs((AX[0,2]-A[ite-1,2])/AX[0,2])
        Err=AX[0,1]
        xr=x0-(fun(x0)[0]/fun(x0)[1])
        x0=xr
        A=np.append(A,AX,axis=0)
        ite+=1
    return A

## METODO SECANTE ##
def seca(fun,x0,x00,es):
    ite=0
    Err=1
    xr=x0
    A=np.empty((0,3),dtype=np.float64)
    while Err>es:
        AX=np.zeros((1,3),dtype=np.float64)
        AX[0,0]=ite+1
        AX[0,2]=xr
        if ite==0:
            AX[0,1]=1
        else:
            AX[0,1]=abs((AX[0,2]-A[ite-1,2])/AX[0,2])
        Err=AX[0,1]
        xr=x0-(fun(x0)[0]*(x00-x0)/(fun(x00)[0]-fun(x0)[0]))
        x00=x0
        x0=xr
        A=np.append(A,AX,axis=0)
        ite+=1
    return A


####################### SEMANA 3 ############################
#		   METODOS DIRECTOS
#############################################################

##FORWARD##
def forward(M,Y):
    n=len(M)
    X=np.zeros((n,1),dtype=np.float64)
    for i in range(n):
        sp=0
        for j in range(n):
            if j<i:
                sp+=M[i,j]*X[j]
        X[i]=(Y[i]-sp)/M[i,i]
    return X

##BACKWARD##
def backward(M,Y):
    n=len(M)
    X=np.zeros((n,1),dtype=np.float64)
    for i in reversed(range(n)):
        sp=0
        for j in reversed(range(n)):
            if j>i:
                sp+=M[i,j]*X[j]
        X[i]=(Y[i]-sp)/M[i,i]
    return X        # vcol N

##ELIMINACION GAUSS##
def elgaU(M,Y): #into matriz triangular superior c/diag 1
    n=len(M)
    Ma=np.append(M,Y,axis=1)
    for i in range(n):
        val=Ma[i,i]
        Ma[i]=Ma[i]/val
        for j in range(i+1,n):
            factor=Ma[j,i]
            if factor!=0:
                Ma[j]-=Ma[i]*factor
    Mr=np.zeros((n,n),dtype=np.float64)
    for k in range(n):
        Mr[:,k]=Ma[:,k]
    Yr=np.resize(Ma[:,n],(n,1))
    return Mr,Yr


def elgaL(M,Y): #into matriz triangular inferior c/diag 1
    n=len(M)
    Ma=np.append(M,Y,axis=1)
    for i in reversed(range(n)):
        val=Ma[i,i]
        Ma[i]=Ma[i]/val
        for j in reversed(range(i)):
            factor=Ma[j,i]
            if factor!=0:
                Ma[j]-=Ma[i]*factor
    Mr=np.zeros((n,n),dtype=np.float64)
    for k in range(n):
        Mr[:,k]=Ma[:,k]
    Yr=np.resize(Ma[:,n],(n,1))
    return Mr,Yr


## PIVOTEO PARCIAL C/ADICIONAL ##
def pivop(M,Y):
    n=len(M)
    P=np.resize(np.arange(n),(n,1))
    Ma=np.append(M,Y,axis=1)
    Ma=np.append(Ma,P,axis=1)
    for i in range(n):
        ufmax=np.where(abs(Ma[i:,i])==np.amax(abs(Ma[i:,i])))[0]
        if len(ufmax)==1:
            ufmax+=i
            if ufmax!=i:
                uu=Ma[ufmax]
                Ma[ufmax]=Ma[i]
                Ma[i]=uu
            else:
                pass
    for j in range(n):
        if Ma[j,j]==0:
            uf=np.amin(np.where(abs(Ma[:,i])==np.amax(abs(Ma[:,i])))[0])
            Ma[j,:n]+=Ma[uf,:n]
    for k in range(n):          
        M[:,k]=Ma[:,k]
    Y=np.resize(Ma[:,n],(n,1))
    A=np.resize(Ma[:,n+1],(n,1))
    return M,Y,A,Ma


##LU CROUT##
def lucroutU(M,Y):  # M,Y --> U --> L --> X
    n=len(M)
    U=elgaU(M,Y)[0]
    L=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(i+1):
            ss=np.inner(L[i,:],U[:,j])
            L[i,j]=M[i,j]-ss
    d=forward(L,Y)
    X=backward(U,d) 
    return X,L,U

def lucroutL(M,Y):  # M,Y --> L --> U --> X
    n=len(M)
    L=elgaL(M,Y)[0]
    U=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(i+1):
            ss=sum(U[k,i]*L[j,k] for k in range(j))
            U[j,i]=M[j,i]-ss
        for j in range(i,n):
            sr=sum(U[k,i]*L[j,k] for k in range(i))
            L[j,i]=(M[j,i]-sr)/U[i,i]
    d=forward(L,Y)
    X=backward(U,d)
    return X,L,U

##CHOLESKY##
def cholesky(M,Y):
    n=len(M)
    L=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for k in range(i+1):
            ss=sum(L[i,j]*L[k,j] for j in range(k))
            if i==k:
                L[i,k]=(M[i,i]-ss)**.5
            else:
                L[i,k]=(M[i,k]-ss)/L[k,k]
    d=forward(L,Y)
    X=backward(L.T,d)
    return X,L,L.T


####################### SEMANA 4 ############################
#		   METODOS ITERATIVOS
#############################################################

## JACOBI ##
def jaco(A,b,e):  # matriz NxN, vector columna, error
    n=len(A)
    X1=np.zeros((n,1),dtype=np.float64)
    X2=np.zeros((n,1),dtype=np.float64)
    XJ=np.empty((0,2),dtype=np.float64)
    xj=np.zeros((1,2),dtype=np.float64)
    Jer=1
    ite=1
    while Jer>e:
        jj=0
        for i in range(n):
            sJ=0
            for j in range(n):
                if j!=i:
                    sJ+=A[i,j]*X2[j]
            X1[i]=(b[i]-sJ)/A[i,i]
        Jer=np.linalg.norm(X1-X2)/np.linalg.norm(X1)
        xj[0,0]=ite
        xj[0,1]=Jer
        XJ=np.append(XJ,xj,axis=0)
        ite+=1
        X2=np.copy(X1)
    return X1,XJ,ite

## GAUSS-SEIDEL ##
def gause(A,b,e):
    n=len(A)
    X1=np.zeros((n,1),dtype=np.float64)
    X2=np.zeros((n,1),dtype=np.float64)
    XG=np.empty((0,2),dtype=np.float64)
    xg=np.zeros((1,2),dtype=np.float64)
    Ger=1
    ite=1
    while Ger>e:
        for i in range(n):
            sG=0
            for j in range(n):
                if j<i:
                    sG+=A[i,j]*X1[j]
                if j>i:
                    sG+=A[i,j]*X2[j]
            X1[i]=(b[i]-sG)/A[i,i]
        Ger=np.linalg.norm(X1-X2)/np.linalg.norm(X1)
        xg[0,0]=ite
        xg[0,1]=Ger
        XG=np.append(XG,xg,axis=0)
        ite+=1
        X2=np.copy(X1)
    return X1,XG,ite

## SOR ##
def sora(A,b,w,e):
    n=len(A)
    X1=np.zeros((n,1),dtype=np.float64)
    X2=np.zeros((n,1),dtype=np.float64)
    XS=np.empty((0,2),dtype=np.float64)
    xs=np.zeros((1,2),dtype=np.float64)
    Ser=1
    ite=1
    while Ser>e:
        for i in range(n):
            sS=0
            for j in range(n):
                if j<i:
                    sS+=w*A[i,j]*X1[j]
                    sS+=(1-w)*A[i,j]*X2[j]
                if j>i:
                    sS+=A[i,j]*X2[j]
            X1[i]=(b[i]-sS)/A[i,i]
        Ser=np.linalg.norm(X1-X2)/np.linalg.norm(X1)
        xs[0,0]=ite
        xs[0,1]=Ser
        XS=np.append(XS,xs,axis=0)
        ite+=1
        X2=np.copy(X1)
    return X1,XS,ite

## MÁXIMO DESCENSO ##
def made(A,b,e):
    n=len(A)
    X1=np.zeros((n,1),dtype=np.float64)
    X2=np.zeros((n,1),dtype=np.float64)
    XS=np.empty((0,3),dtype=np.float64)
    R=A.dot(X1)-b
    Er=1
    ite=0
    while Er>e:
        alpha=(R.T).dot(R)/(R.T).dot(A.dot(R))
        X2=X1-alpha*R
        Er=np.linalg.norm(X2-X1)
        X1=np.copy(X2)
        R=A.dot(X1)-b
        XS=np.append(XS,np.array([[ite,Er,alpha]]),axis=0)
        ite+=1
    return X1.T,XS,ite

## GRADIENTE CONJUGADO ##
def graco(A,b,e):
    n=len(A)
    X1=np.zeros((n,1),dtype=np.float64)
    X2=np.zeros((n,1),dtype=np.float64)
    XS=np.empty((0,4),dtype=np.float64)
    R=A.dot(X1)-b
    P=-R
    Er=1
    ite=0
    while Er>e:
        alpha=(R.T).dot(R)/(P.T).dot(A.dot(P))
        X2=X1-alpha*P
        Rk1=R+alpha*A.dot(P)
        beta=(Rk1.T).dot(Rk1)/(R.T).dot(R)
        Pk1=-Rk1+beta*P
        Er=np.linalg.norm(X2-X1)
        R=np.copy(Rk1)
        P=np.copy(Pk1)
        X1=np.copy(X2)
        XS=np.append(XS,np.array([[ite,Er,alpha,beta]]),axis=0)
        ite+=1
    return X1,XS,ite	


############################ SEMANA 5 ###############################
#		INTERPOLACION POLINOMIAL - PARTE 1
#####################################################################

## MONOMIO ##
def mono(M,x):
    n=len(M)
    mM=np.zeros((n,n),dtype=np.float64)
    mY=np.zeros((n,1),dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mM[i,j]=M[i,0]**j
        mY[i]=M[i,1]
    A=elgaU(mM,mY)
    B=backward(A[0],A[1])
    mX=sum(B[k]*x**k for k in range(n))
    return B.T,mX

## NEWTON ##
def newt(M,x):
    n=len(M)
    nM=np.zeros((n,n),dtype=np.float64)
    nY=np.zeros((n,1),dtype=np.float64)
    for i in range(n):
        for j in range(n):  #keyword: newton basis
            ss=1
            if j<=i:
                for k in range(j):
                    ss*=(M[i,0]-M[k,0]) #pi_0=1 , pi_j=prod._k=0,...j-1 {t_i-t_k}
                nM[i,j]=ss
        nM[i,0]=1
        nY[i]=M[i,1]
    F=forward(nM,nY)
    A=np.ones((n,n),dtype=np.float64)
    for i in range(n-1,0,-1):
        for j in range(n-1,0,-1):
            val=x-M[j-1,0]
            if j<=i:
                A[i,j]=val
    nX=sum(np.prod(A[k,:])*F[k] for k in range(n))
    return F.T,nX


## LAGRANGE ##
def lagra(M,x):
    n=len(M)
    C=np.zeros((n,1),dtype=np.float64)
    for i in range(n):
        sx=1
        for j in range(n):
            if i!=j:
                sx*=(x-M[j,0])/(M[i,0]-M[j,0])
        C[i]=sx
    lX=sum(C[k]*M[k,1] for k in range(n))
    return C.T,lX


## PUNTOS CHEBISHEV ##
def chebpoint(fun,a,b,n):
    M=np.zeros((n,2),dtype=np.float64)
    for i in range(n):
        M[i,0]=a+(b-a)*(1-np.cos(i*np.pi/(n-1)))/2.0
        M[i,1]=fun(M[i,0])
    return M


## PUNTOS EQUIDISTANTES ##
def eqpoint(fun,a,b,n):
    M=np.zeros((n,2),dtype=np.float64)
    for i in range(n):
        M[i,0]=a+(b-a)*i/(n-1)
        M[i,1]=fun(M[i,0])
    return M


##   GRAFICA EQUIDISTANTES - LAGRANGE    ##
def geqla(fun,a,b,n):
    M=eqpoint(fun,a,b,n)
    N=np.zeros((n,3),dtype=np.float64)	# puntos X, puntos Y, f(x)-Y.estimado.lagra
    for i in range(n):          # puntos para graficar
        N[i,0]=a+(b-a)*(i+1e-6)/(n-1)
        N[0,0]=0.0
        N[n-1,0]=1.0
    for j in range(n):
        N[j,1]=lagra(M,N[j,0])[1]
    for k in range(n):
        N[k,2]=abs(fun(N[k,0])-N[k,1])
    return N


##   GRAFICA CHEBYSHEV - LAGRANGE    ##
def gxebla(fun,a,b,n):
    M=chebpoint(fun,a,b,n)      # puntos para formar función polinomial
    N=np.zeros((n,3),dtype=np.float64)	# puntos X, puntos Y, f(x)-Y.estimado.lagra
    for i in range(n):          # puntos para graficar
        N[i,0]=a+(b-a)*(i+1e-6)/(n-1)
        N[0,0]=0.0
        N[n-1,0]=1.0
    for j in range(n):
        N[j,1]=lagra(M,N[j,0])[1]
    for k in range(n):
        N[k,2]=abs(fun(N[k,0])-N[k,1])
    return N


############################ SEMANA 6 ###############################
#		INTERPOLACION POLINOMIAL - PARTE 2
#####################################################################


##      SPLINE-1        ##
def spl1(M,x):
    n=len(M)-1
    M=np.append(M,np.zeros((n+1,1)),axis=1)
    for i in range(n):
        M[i,2]=(M[i+1,1]-M[i,1])/(M[i+1,0]-M[i,0])
    ss=0
    for j in range(n):
        if x>=M[j,0] and x<M[j+1,0]:
            ss=M[j,1]+M[j,2]*(x-M[j,0])
        elif x==M[n,0]:
            ss=M[n-1,1]+M[n-1,2]*(x-M[n-1,0])
    return ss,M


##      SPLINE-2        ##
def spl2(M,x):
    n=len(M)-1
    N=np.zeros((3*n,3*n),dtype=np.float64)
    b=np.zeros((3*n,1),dtype=np.float64)
    #continuidad C0 y extremos:
    for i in range(n):
        for j in range(3):
            N[i,j+3*i]=M[i,0]**(2-j)
        b[i]=M[i,1]
    for i in range(n):
        for j in range(3):
            N[i+n,j+3*i]=M[i+1,0]**(2-j)
        b[i+n]=M[i+1,1]
    #continuidad C1:
    for i in range(n-1):
        for j in range(3):
            if j%3==0:
                N[i+2*n,j+3*i]=2*M[i+1,0]
                N[i+2*n,j+3*i+3]=-2*M[i+1,0]
            if j%3==1:
                N[i+2*n,j+3*i]=1
                N[i+2*n,j+3*i+3]=-1
    #inicio C2:
    N[3*n-1,0]=1
    #solver sys.lin.eq
#    P=pivop(N,b)
#    UP=lucroutL(P[0],P[1])[0]
    UP=np.linalg.solve(N,b)
    UP.resize(n,3)
    ss=0
    for k in range(n):
        if x>=M[k,0] and x<M[k+1,0]:
            ss=UP[k,0]*x**2+UP[k,1]*x+UP[k,2]
        elif x==M[n,0]:
            ss=UP[n-1,0]*x**2+UP[n-1,1]*x+UP[n-1,2]
    return ss,N,UP


##      SPLINE-3        ##
def spl3(M,x):
    n=len(M)-1
    N=np.zeros((4*n,4*n),dtype=np.float64)
    b=np.zeros((4*n,1),dtype=np.float64)
    #continuidad C0 y extremos:
    for i in range(n):
        for j in range(4):
            N[i,j+4*i]=M[i,0]**(3-j)
        b[i]=M[i,1]
    for i in range(n):
        for j in range(4):
            N[i+n,j+4*i]=M[i+1,0]**(3-j)
        b[i+n]=M[i+1,1]
    #continuidad C1:
    for i in range(n-1):
        for j in range(4):
            if j%4==0:
                N[i+2*n,j+4*i]=3*M[i+1,0]**2
                N[i+2*n,j+4*i+4]=-3*M[i+1,0]**2
            if j%4==1:
                N[i+2*n,j+4*i]=2*M[i+1,0]
                N[i+2*n,j+4*i+4]=-2*M[i+1,0]
            if j%4==2:
                N[i+2*n,j+4*i]=1
                N[i+2*n,j+4*i+4]=-1
    #continuidad C2:
    for i in range(n-1):
        for j in range(4):
            if j%4==0:
                N[i+3*n-1,j+4*i]=6*M[i+1,0]
                N[i+3*n-1,j+4*i+4]=-6*M[i+1,0]
            if j%4==1:
                N[i+3*n-1,j+4*i]=2
                N[i+3*n-1,j+4*i+4]=-2
    #extremos C3:
    N[4*n-1,0]=1
    N[4*n-2,4*n-4]=1
    #solver sys.lin.eq
#    P=pivop(N,b)
#    UP=lucroutL(P[0],P[1])[0]
    UP=np.linalg.solve(N,b)
    UP.resize(n,4)
    ss=0
    for k in range(n):
        if x>=M[k,0] and x<M[k+1,0]:
            ss=UP[k,0]*x**3+UP[k,1]*x**2+UP[k,2]*x+UP[k,3]
        elif x==M[n,0]:
            ss=UP[n-1,0]*x**3+UP[n-1,1]*x**2+UP[n-1,2]*x+UP[n-1,3]
    return ss,N,UP


####################### SEMANA 7 ############################
#	   	   AJUSTE DE CURVAS
#############################################################


## METODOS LINEALIZADOS GENERICO - FUNCION 1 VARIABLE ##
def lingen(M,poly):	#matriz datos, tipo polinomio a elegir
    m=len(M)
    n=len(poly(M[0,0]))
    A=np.zeros((m,n),dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A[i,j]=poly(M[i,0])[j]
    B=np.matmul(A.T,A)
    C=np.matmul(A.T,M[:,1])
    XX=np.linalg.solve(B,C)
#    XX=cholesky(B,C)[0]
    St=sum((M[k,1]-(np.sum(M[:,1])/m))**2 for k in range(m))
    Sr=sum((M[k,1]-np.dot(A[k,:],XX))**2 for k in range(m))
    PP=np.zeros((2,1),dtype=np.float64)
    PP[0]=((St-Sr)/St)  #R**2
    PP[1]=(Sr/(m-n-1))**.5  #Sy/x
    return XX,PP

def pl0(x):
    return np.array([[1]],dtype=np.float64).T

# POLY 1 - FUNCION 1 VARIABLE #
def pl1(x):
    return np.array([[1,x]],dtype=np.float64).T

# POLY 2 - FUNCION 1 VARIABLE #
def pl2(x):
    return np.array([[1,x,x**2]],dtype=np.float64).T

## METODOS NO LINEALIZADOS GENERICO - FUNCION 1 VARIABLE ##
def nolingen(M,n,dfun,err): #matriz datos, N°parametros, funcion, error
    m=len(M)
    A=np.ones((n,1),dtype=np.float64)
    AA=np.ones((n,1),dtype=np.float64)
    F=np.zeros((m,1),dtype=np.float64)
    e=1
    ite=0
    while e>err:
        Z=np.zeros((m,n),dtype=np.float64)
        D=np.copy(M[:,1]).reshape(m,1)
        for i in range(m):
            for j in range(n):
                Z[i,j]=dfun(A,M[i,0])[j]
            F[i]=dfun(A,M[i,0])[n]
        ZZ=np.matmul(Z.T,Z)
        D-=F
        DA=np.linalg.solve(ZZ,np.matmul(Z.T,D))
#        DA=lucroutL(ZZ,np.matmul(Z.T,D))[0]
        A+=DA
        AA=np.append(AA,A,axis=1)
        e=np.linalg.norm(DA)
        ite+=1
    St=sum((M[k,1]-(np.sum(M[:,1])/m))**2 for k in range(m))
    Sr=sum((M[k,1]-F[k])**2 for k in range(m))
    Sz=sum((F[k]-(np.sum(M[:,1])/m))**2 for k in range(m))
    PP=np.zeros((2,1),dtype=np.float64)
    PP[0]=((St-Sr)/St)  #R**2
    PP[1]=(Sr/(m-n-1))**.5  #Sy/x
    return A,PP


## METODOS LINEALIZADOS - FUNCION 2 VARIABLES ##
def xlingen(M,poly):
    m=len(M)
    n=len(poly(M[0,0],M[0,1]))
    A=np.zeros((m,n),dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A[i,j]=poly(M[i,0],M[i,1])[j]
    B=np.matmul(A.T,A)
    C=np.matmul(A.T,M[:,n-1])
    XX=mtd.cholesky(B,C)[0]
    St=sum((M[k,n-1]-(np.sum(M[:,n-1])/m))**2 for k in range(m))
    Sr=sum((M[k,n-1]-np.dot(A[k,:],XX))**2 for k in range(m))
    PP=np.zeros((2,1),dtype=np.float64)
    PP[0]=((St-Sr)/St)  #R**2
    PP[1]=(Sr/(m-n-1))**.5  #Sy/x
    return XX,PP

# POLY 1 - FUNCION 2 VARIABLES #
def xpl1(x,y):
    return np.array([[1,x,y]],dtype=np.float64).T


####################### SEMANA 8 ############################
#	   	   INTEGRACION - PARTE 1
#############################################################

# METODO TRAPECIO SIMPLE
def itraps(fun,a,b):
    return (b-a)*(fun(a)+fun(b))/2

# METODO TRAPECIO COMPUESTO
def itrapc(fun,a,b,n):
    h=(b-a)/n
    return h/2*(fun(a)+fun(b)+2*sum(fun(a+h*i) for i in range(1,n)))

# METODO SIMPSON 1/3 SIMPLE
def ispn13s(fun,a,b):
    return (b-a)/6*(fun(a)+fun(b)+4*fun((a+b)/2))

# METODO SIMPSON 1/3 COMPUESTO
def ispn13c(fun,a,b,n):
    h=(b-a)/n
    suma=0
    for i in range(1,n):
        if i%2==0:
            suma+=2*fun(a+h*i)
        else:
            suma+=4*fun(a+h*i)
    return h/3*(fun(a)+fun(b)+suma)

# METODO SIMPSON 3/8 SIMPLE
def ispn38s(fun,a,b):
    return (b-a)/8*(fun(a)+fun(b)+3*fun((b+2*a)/3)+3*fun((2*b+a)/3))

# METODO SIMPSON 3/8 COMPUESTO
def ispn38c(fun,a,b,n):
    h=(b-a)/n
    suma=0
    for i in range(1,n):
        if i%3==0:
            suma+=2*fun(a+h*i)
        else:
            suma+=3*fun(a+h*i)
    return h*3/8*(fun(a)+fun(b)+suma)


# METODO RICHARDSON
def irichardson(fun,a,b,n,m):
    return itrapc(fun,a,b,n)+(itrapc(fun,a,b,n)-itrapc(fun,a,b,m))/((n/m)**2-1)


####################### SEMANA 9 ############################
#	   	   INTEGRACION/DIF - PARTE 2
#############################################################

# CUADRATURA GAUSS-LEGENDRE
def qGL(fun,a,b,N,x):
    if N==1:
        xi=[0,0]
        wi=[2,0]
    elif N==2:
        xi=[1/3**.5,-1/3**.5]
        wi=[1,1]
    elif N==3:
        a3=(3/5)**.5
        xi=[0,a3,-a3]
        wi=[8/9,5/9,5/9]
    elif N==4:
        a4=((3/7)-(2/7)*(6/5)**.5)**.5
        b4=((3/7)+(2/7)*(6/5)**.5)**.5
        c4=(18+30**.5)/36
        d4=(18-30**.5)/36
        xi=[a4,-a4,b4,-b4]
        wi=[c4,c4,d4,d4]
    elif N==5:
        a5=(5-2*(10/7)**.5)**.5/3
        b5=(5+2*(10/7)**.5)**.5/3
        c5=(322+13*70**.5)/900
        d5=(322-13*70**.5)/900
        xi=[0,a5,-a5,b5,-b5]
        wi=[128/225,c5,c5,d5,d5]
    k1=(b-a)/2
    k2=(b+a)/2
    return k1*sum(fun(k1*xi[j]+k2,x)*wi[j] for j in range(len(xi)))

# CUADRATURA GAUSS-RADAU-LEGENDRE
def qGRL(fun,a,b,N):
    if N==2:
        xi=[-1,1/2]
        wi=[1/3,3/2]
    elif N==3:
        xi=[-1,1/5*(1-6**.5),1/5*(1+6**.5)]
        wi=[2/9,(16+6**.5)/18,(16-6**.5)/18]
    elif N==4:
        xi=[-1,-.575319,.181066,.822824]
        wi=[2/16,.657689,.776387,.440924]
    elif N==5:
        xi=[-1,-.720480,-.167181,.446314,.885792]
        wi=[2/25,.446208,.623653,.562712,.287427]
    k1=(b-a)/2
    k2=(b+a)/2
    return k1*sum(fun(k1*xi[j]+k2)*wi[j] for j in range(N))

# CUADRATURA GAUSS-LOBATTO-LEGENDRE
def qGLL(fun,a,b,N):
    if N==2:
        xi=[0,1]
        wi=[0,1]
    elif N==3:
        xi=[0,1]
        wi=[4/3,1/3]
    elif N==4:
        xi=[0,.447213,1]
        wi=[0,2.5/3,.5/3]
    elif N==5:
        xi=[0,.654653,1]
        wi=[64/90,4.9/9,.1]
    elif N==6:
        xi=[0,.285231,.765055,1]
        wi=[0,.554858,.378474,.2/3]
    elif N==7:
        xi=[0,.468848,.830223,1]
        wi=[.487619,.431745,.276826,.047619]
    elif N==8:
        xi=[0,.209299,.591700,.871740,1]
        wi=[0,.412458,.341122,.210704,.035714]
    k1=(b-a)/2
    k2=(b+a)/2
    return k1*(sum((fun(k1*xi[j]+k2)+fun(-k1*xi[j]+k2))*wi[j] for j in range(len(xi)))-fun(k1*xi[0]+k2)*wi[0])

# DIFERENCIACION FORWARD
def dforw(M,v,p):
    h=M[0,p+1]-M[0,p]
    fd1=np.dot(M[v,p:p+2],[-1,1])/h
#    fd2=np.dot(M[v,p:p+2+2],[-3,4,-1])/2/h
    sd1=np.dot(M[v,p:p+3],[1,-2,1])/h/h
#    sd2=np.dot(M[v,p:p+3+2],[2,-5,4,-1])/h/h
    return fd1,sd1

# DIFERENCIACION BACKWARD
def dback(M,v,p):
    h=M[0,p]-M[0,p-1]
    fd1=np.dot(M[v,p-1:p+1],[-1,1])/h
#    fd2=np.dot(M[v,p-2:p+0+1],[1,-4,3])/2/h
    sd1=np.dot(M[v,p-2:p+1],[1,-2,1])/h/h
#    sd2=np.dot(M[v,p-3:p+0+1],[-1,4,-5,2])/h/h
    return fd1,sd1

# DIFERENCIACION CENTRAL
def dcent(M,v,p):
    h=M[0,p]-M[0,p+1]
    fd1=np.dot(M[v,p-1:p+2],[1,0,-1])/2/h
#    fd2=np.dot(M[v,p-2:p+2+1],[-1,8,0,-8,1])/12/h
    sd1=np.dot(M[v,p-1:p+2],[1,-2,1])/h/h
#    sd2=np.dot(M[v,p-2:p+2+1],[-1,16,-30,16,-1])/12/h/h
    return fd1,sd1

####################### SEMANA 10 ############################
#	   	   	  EDO´s
#############################################################

# METODO EULER - 1 VAR DEP / 1 VAR INDEP
def edo_euler(fun,xi,yi,h,xf):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        ny=yi+fun(xi,yi)*h
        yi=ny
        xi+=h
    return ny

# METODO HEUN - 1 VAR DEP / 1 VAR INDEP
def edo_heun(fun,xi,yi,h,xf,ee):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        e=1
        ff=fun(xi,yi)
        fny=yi+ff*h
        while e>ee:
            ny=yi+(ff+fun(xi+h,fny))*(h/2)
            e=abs((ny-fny)/ny)
            fny=ny
        yi=ny
        xi+=h
    return ny

# METODO RUNGE_KUTTA 2 - 1 VAR DEP / 1 VAR INDEP (1/2 HEUN, 2/3 RALSTON, 1 PUNTO MEDIO)
def edo_rk2(fun,xi,yi,h,xf,a2):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        k1=fun(xi,yi)
        k2=fun(xi+h,yi+k1*h)
        ny=yi+((1-a2)*k1+a2*k2)*h
        yi=ny
        xi+=h
    return ny

# METODO RUNGE_KUTTA 3 - 1 VAR DEP / 1 VAR INDEP
def edo_rk3(fun,xi,yi,h,xf):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        k1=fun(xi,yi)
        k2=fun(xi+.5*h,yi+.5*k1*h)
        k3=fun(xi+h,yi-k1*h+2*k2*h)
        ny=yi+(k1+4*k2+k3)*h/6
        yi=ny
        xi+=h
    return ny

# METODO RUNGE_KUTTA 4 - 1 VAR DEP / 1 VAR INDEP
def edo_rk4(fun,xi,yi,h,xf):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        k1=fun(xi,yi)
        k2=fun(xi+.5*h,yi+.5*k1*h)
        k3=fun(xi+.5*h,yi+.5*k2*h)
        k4=fun(xi+h,yi+k3*h)
        ny=yi+(k1+2*k2+2*k3+k4)*h/6
        yi=ny
        xi+=h
    return ny

# METODO RUNGE_KUTTA 5 - 1 VAR DEP / 1 VAR INDEP
def edo_rk5(fun,xi,yi,h,xf):
    step=int((xf-xi)/h)
    ny=yi
    for i in range(step):
        k1=fun(xi,yi)
        k2=fun(xi+h/4,yi+k1*h/4)
        k3=fun(xi+h/4,yi+k1*h/8+k2*h/8)
        k4=fun(xi+h/2,yi-k2*h/2+k3*h)
        k5=fun(xi+h*3/4,yi+k1*h*3/16+k4*h*9/16)
        k6=fun(xi+h,yi-k1*h*3/7+k2*h*2/7+k3*h*12/7-k4*h*12/7+k5*h*8/7)
        ny=yi+(7*k1+32*k3+12*k4+32*k5+7*k6)*h/90
        yi=ny
        xi+=h
    return ny

# METODO EULER - M VAR DEP / 1 VAR INDEP
def medo_euler(mfun,xmyi,h,xf):
    xmyi=np.array(xmyi)
    step=int((xf-xmyi[0])/h)
    i=0
    while i<step:
        xmyn=xmyi+mfun(xmyi)*h
        xmyi=xmyn
        xmyi[0]+=h
        i+=1
    return xmyn

# METODO RUNGE_KUTTA 4 - M VAR DEP / 1 VAR INDEP
def medo_rk4(mfun,xmyi,h,xf):
    xmyi=np.array(xmyi)
    step=int((xf-xmyi[0])/h)
    i=0
    xmyn=xmyi
    xh=xmyi*0
    xh[0]+=h
    while i<step:
        k1=mfun(xmyi)
        k2=mfun(xmyi+.5*xh+.5*k1*h)
        k3=mfun(xmyi+.5*xh+.5*k2*h)
        k4=mfun(xmyi+1.*xh+1.*k3*h)
        xmyn=xmyi+(k1+2*k2+2*k3+k4)*h/6
        xmyi=xmyn
        xmyi[0]+=h
        i+=1
    return xmyn

# METODO EULER IMPLICITO - N VAR DEP
def edo_impeuler(mxfun,yi,h,t):
    yi=np.array([yi[0],yi[1]],dtype=np.float64)
    tt=np.arange(t[0],t[1]+h,h)
    yy=np.zeros((len(tt),len(mxfun)),dtype=np.float64)
    for i in range(len(tt)):
        yy[i]=yi
        mxxfun=-mxfun*h+np.identity(len(mxfun))
        A=np.linalg.solve(mxxfun,yi)
        yi=A
    return yy

# METODO EULER EXPLICITO - N VAR DEP
def edo_expeuler(mxfun,yi,h,t):
    yi=np.array([yi[0],yi[1]],dtype=np.float64)
    tt=np.arange(t[0],t[1]+h,h)
    yy=np.zeros((len(tt),len(mxfun)),dtype=np.float64)
    for i in range(len(tt)):
        mxxfun=mxfun*h+np.identity(len(mxfun))
        yy[i]=np.dot(mxxfun,yi)
        yi=yy[i]
    return yy

#  METODO HEUN MODIFICADO - 1 VAR DEP / 1 VAR INDEP
def edo_modheun(fun,xi,yi,h,xf,error):
    step=int((xf-xi)/h)
    yii=fun(xi-h,yi)    #solo para iniciar
    yn=yi
    for i in range(step):
        e=1
        ff=fun(xi,yi)
        fyn=yii+ff*2*h
        while e>error:
            yn=yi+(ff+fun(xi+h,fyn))/2*h
            e=abs(yn-fyn)/yn
            fyn=yn
        yii=yi
        yi=yn
        xi+=h
    return yn

# FORMULAS ADAM-BASHFORTH
def adam_bash(x):
    M=np.zeros((6,7),dtype=np.float64)
    M[:,0]=[1,2,3,4,5,6]
    M[:,1]=[1,3/2,23/12,55/24,1901/720,4277/720]
    M[:,2]=[0,-1/2,-16/12,-59/24,-2774/720,-7923/720]
    M[:,3]=[0,0,5/12,37/24,2616/720,9982/720]
    M[:,4]=[0,0,0,-9/24,-1274/720,-7298/720]
    M[:,5]=[0,0,0,0,251/720,2877/720]
    M[:,6]=[0,0,0,0,0,-475/720]
    return M

# FORMULAS ADAM-MULTON
def adam_multon(x):
    M=np.zeros((5,7),dtype=np.float64)
    M[:,0]=[2,3,4,5,6]
    M[:,1]=[1/2,5/12,9/24,251/720,475/1440]
    M[:,2]=[1/2,8/12,19/24,646/720,1427/1440]
    M[:,3]=[0,-1/12,-5/24,-264/720,-798/1440]
    M[:,4]=[0,0,1/24,106/720,482/1440]
    M[:,5]=[0,0,0,-19/720,-173/1440]
    M[:,6]=[0,0,0,0,27/1440]
    return M
