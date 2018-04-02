import numpy as np
import math 
import numpy.matlib
import scipy.io as sio
from numpy import linalg as LA
import cvxopt
from cvxopt import solvers
from cvxopt import matrix

##Load Data
mat1 = sio.loadmat('X_train_2.mat')
mat2 = sio.loadmat('Y_train_2.mat')
mat3 = sio.loadmat('X_test_2.mat')
mat4 = sio.loadmat('Y_test_2.mat')
X_train_2=np.asmatrix(mat1['X_final'])
Y_train_2=np.asmatrix(mat2['Y_final'])
X_test_2=np.asmatrix(mat3['X_test'])
Y_test_2=np.asmatrix(mat4['Y_test'])

Y_train_2=np.int32(Y_train_2)
Y_test_2=np.int32(Y_test_2)

#MODIFY DATA FOR SINGLE CLASS
for i in range (0,len(Y_train_2),1):
    if Y_train_2[i]==1:
        Y_train_2[i]=1;        
    else:
        Y_train_2[i]=-1;
for i in range (0,len(Y_test_2),1):
    if Y_test_2[i]==1:
        Y_test_2[i]=1;
    else:
        Y_test_2[i]=-1;

Y_train_2[1500:3000]=0
        
l=np.count_nonzero(Y_train_2)
u=len(Y_train_2)-l
n=l+u

alpha2=np.matlib.zeros((l+u,1))
beta2=np.matlib.zeros((l,1))
K2=np.matlib.zeros((l+u,l+u))
Kx2=np.matlib.zeros((len(X_test_2),l+u))
J2=np.matlib.zeros((l,l+u))
L2=np.matlib.zeros((l+u,l+u))
W2=np.matlib.zeros((l+u,l+u))
D2=np.matlib.zeros((l+u,l+u))
Y_predcted2=np.matlib.zeros((len(X_test_2),1))
Q2=np.matlib.zeros((l,l))
Yd2=np.matlib.zeros((l,l))
f2=np.matlib.zeros((len(X_test_2),1))

sigma2=200
gamma_A2=100
gamma_I2=10 #gamma_I2=0 for supervised


for i in range(0,l,1):
    J2[i,i]=1
    Yd2[i,i]=Y_train_2[i]
for i in range(0,n,1):
    xi=X_train_2[i]
    for j  in range(0,n,1):
        xj=X_train_2[j]
        K2[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma2**2)))         
        W2[i,j]=np.exp(-((LA.norm(xi-xj))**2))                              
d=np.sum(W2,axis=1)
for i in range(0,l+u,1):
    D2[i,i]=d[i]
L2=D2-W2



#LOAD DATA
mat1 = sio.loadmat('X_train.mat')
mat3 = sio.loadmat('X_test.mat')
X_train_1=np.asmatrix(mat1['X_final'])
Y_train_1=Y_train_2
X_test_1=np.asmatrix(mat3['X_test'])
Y_test_1=Y_test_2



alpha1=np.matlib.zeros((l+u,1))
beta1=np.matlib.zeros((l,1))
K1=np.matlib.zeros((l+u,l+u))
Kx1=np.matlib.zeros((len(X_test_1),l+u))
J1=np.matlib.zeros((l,l+u))
L1=np.matlib.zeros((l+u,l+u))
W1=np.matlib.zeros((l+u,l+u))
D1=np.matlib.zeros((l+u,l+u))
Y_predcted1=np.matlib.zeros((len(X_test_1),1))
Q1=np.matlib.zeros((l,l))
Yd1=np.matlib.zeros((l,l))
f1=np.matlib.zeros((len(X_test_1),1))

sigma1=0.05
gamma_A1=100
gamma_I1=10



for i in range(0,l,1):
    J1[i,i]=1
    Yd1[i,i]=Y_train_1[i]
for i in range(0,n,1):
    xi=X_train_1[i]
    for j  in range(0,n,1):
        xj=X_train_1[j]
        K1[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma1**2)))         
        W1[i,j]=np.exp(-((LA.norm(xi-xj))**2))                             
d=np.sum(W1,axis=1)
for i in range(0,l+u,1):
    D1[i,i]=d[i]
L1=D1-W1

alpha_multi=0.5
L=(1-alpha_multi)*L1+(alpha_multi)*L2


Q2=Yd2*J2*K2*(LA.inv((2*gamma_A2*np.eye(n))+((2*gamma_I2/(n^2))*L*K2)))*np.transpose(J2)*Yd2
Q1=Yd1*J1*K1*(LA.inv((2*gamma_A1*np.eye(n))+((2*gamma_I1/(n^2))*L*K1)))*np.transpose(J1)*Yd1


y=(Y_train_2[0:1500])
y=y.astype(np.double)
P = matrix(Q2)
q = -np.ones((l, 1))
q = q.astype(np.double)
cvx_q = matrix(q)
G = matrix(-np.eye(l))
h = matrix(np.zeros(l))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))
solvers.options['show_progress'] = False
sol = solvers.qp(P, cvx_q, G, h, A, b)
beta2 = np.array(sol['x'])
alpha2=(LA.inv((2*gamma_A2*np.eye(n))+((2*gamma_I2/(n^2))*L*K2)))*np.transpose(J2)*Yd2*beta2 

P = matrix(Q1)
solvers.options['show_progress'] = False
sol = solvers.qp(P, cvx_q, G, h, A, b)
beta1 = np.array(sol['x'])
alpha1=(LA.inv((2*gamma_A1*np.eye(n))+((2*gamma_I1/(n^2))*L*K1)))*np.transpose(J1)*Yd1*beta1 


#Testing data for View 1
for i in range(0,len(X_test_1)):
    f=0
    for j in range(0,l+u):
        Kx=np.exp(-((X_test_1[i]-X_train_1[j])*np.transpose(X_test_1[i]-X_train_1[j]))/((2*(sigma1**2))))
        f=f + (alpha1[j]*Kx) 
    f1[i]=f    
              
## Testing Data for View 2
for i in range(0,len(X_test_2)):
    f=0
    for j in range(0,l+u):
        Kx=np.exp(-((X_test_2[i]-X_train_2[j])*np.transpose(X_test_2[i]-X_train_2[j]))/((2*(sigma2**2))))
        f=f + (alpha2[j]*Kx) 
    f2[i]=f  

n_crrct=0    
for i in range(0,len(X_test_2)):
    if abs(f1[i])>abs(f2[i]):
        f=f1[i]
    else:
        f=f2[i]
    if f<=0:    
        Y_predcted2[i]=-1
    else:
        Y_predcted2[i]=1
    if Y_predcted2[i]==Y_test_2[i]:
        n_crrct=n_crrct+1             


Percnt_crrct=100*n_crrct/float(len(X_test_1))       
print(Percnt_crrct) 
