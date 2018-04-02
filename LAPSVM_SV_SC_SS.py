import numpy as np
import math 
import numpy.matlib
import scipy.io as sio
from numpy import linalg as LA
import cvxopt
from cvxopt import solvers
from cvxopt import matrix

## Load Data
mat1 = sio.loadmat('X_train.mat')
mat2 = sio.loadmat('Y_train.mat')
mat3 = sio.loadmat('X_test.mat')
mat4 = sio.loadmat('Y_test.mat')
X_train=np.asmatrix(mat1['X_final'])
Y_train=np.asmatrix(mat2['Y_final'])
X_test=np.asmatrix(mat3['X_test'])
Y_test=np.asmatrix(mat4['Y_test'])

Y_train=np.int32(Y_train)
Y_test=np.int32(Y_test)

sigma=[0.05]
accuracy=np.zeros(len(sigma))
for i in range (0,len(Y_train),1):
    if Y_train[i]==1:
        Y_train[i]=1;            
    else:
        Y_train[i]=-1;
for i in range (0,len(Y_test),1):
    if Y_test[i]==1:
        Y_test[i]=1;
    else:
        Y_test[i]=-1;

Y_train[1500:3000]=0
l=np.count_nonzero(Y_train)
u=len(Y_train)-l
n=l+u
for s in range(0,len(sigma),1):
    alpha=np.matlib.zeros((l+u,1))
    beta=np.matlib.zeros((l,1))
    K=np.matlib.zeros((l+u,l+u))
    Kx=np.matlib.zeros((len(X_test),l+u))
    J=np.matlib.zeros((l,l+u))
    L=np.matlib.zeros((l+u,l+u))
    W=np.matlib.zeros((l+u,l+u))
    D=np.matlib.zeros((l+u,l+u))
    Y_predcted=np.matlib.zeros((len(X_test),1))
    Q=np.matlib.zeros((l,l))
    Yd=np.matlib.zeros((l,l))
    f=np.matlib.zeros((len(X_test),1))


    gamma_A=100
    gamma_I=10 #gamma_I=0 for supervised



    for i in range(0,l,1):
        J[i,i]=1
        Yd[i,i]=Y_train[i]
    for i in range(0,n,1):
        xi=X_train[i]
        for j  in range(0,n,1):
            xj=X_train[j]
            K[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma[s]**2)))         
            W[i,j]=np.exp(-((LA.norm(xi-xj))**2))                              
    d=np.sum(W,axis=1)
    for i in range(0,l+u,1):
        D[i,i]=d[i]
    L=D-W

    Q=Yd*J*K*(LA.inv((2*gamma_A*np.eye(n))+((2*gamma_I/(n^2))*L*K)))*np.transpose(J)*Yd


    y=(Y_train[0:l])
    y=y.astype(np.double)

    P = matrix(Q)
    q = -np.ones((l, 1))
    q = q.astype(np.double)
    cvx_q = matrix(q)

    G = matrix(-np.eye(l))
    h = matrix(np.zeros(l))
    A = matrix(y.reshape(1, -1))




    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, cvx_q, G, h, A, b)
    beta = np.array(sol['x'])



    alpha=(LA.inv((2*gamma_A*np.eye(n))+((2*gamma_I/(n^2))*L*K)))*np.transpose(J)*Yd*beta 

    n_crrct=0
    for i in range(0,len(X_test)):
        
        xi=X_test[i]
        for j in range(0,l+u):
            xj=X_train[j]
            Kx[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma[s]**2)))
            f[i]=f[i] + (alpha[j]*Kx[i,j])  
        if f[i] >= 0:
            Y_predcted[i]=1
        else:
            Y_predcted[i]=-1
        if Y_predcted[i]==Y_test[i]:              
            n_crrct=n_crrct+1
            
    accuracy[s]=100*n_crrct/len(X_test)  
