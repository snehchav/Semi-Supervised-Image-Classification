
import math 
import numpy as np
import numpy.matlib
import scipy.io as sio
from numpy import linalg as LA
import cvxopt
from cvxopt import solvers
from cvxopt import matrix

## Load Data feature set 2
mat1 = sio.loadmat('X_train_2.mat')
mat2 = sio.loadmat('Y_train_2.mat')
mat3 = sio.loadmat('X_test_2.mat')
mat4 = sio.loadmat('Y_test_2.mat')
X2m=np.asmatrix(mat1['X_final'])
Y_train_2=np.asmatrix(mat2['Y_final'])
X_test_2=np.asmatrix(mat3['X_test'])
Y_test_2=np.asmatrix(mat4['Y_test'])
Y_train_2=np.int32(Y_train_2)
Y_test_2=np.int32(Y_test_2)

#Load Data feature set 1
mat1 = sio.loadmat('X_train.mat')
mat2 = sio.loadmat('Y_train.mat')
mat3 = sio.loadmat('X_test.mat')
mat4 = sio.loadmat('Y_test.mat')
X1m=np.asmatrix(mat1['X_train'])
Y_train_1=np.asmatrix(mat2['Y_train'])
X_test_1=np.asmatrix(mat3['X_test'])
Y_test_1=np.asmatrix(mat4['Y_test'])

no_label=6
n_crrct=0
f_final1=np.zeros((len(X_test_1),no_label-1,no_label-1))
f_final2=np.zeros((len(X_test_2),no_label-1,no_label-1))
alpha_final1 = np.zeros((160,no_label-1,no_label-1))
alpha_final2 = np.zeros((160,no_label-1,no_label-1))
Y_predcted=np.matlib.zeros((len(X_test_1),1))

for label in range(1,no_label):
    for label_2 in range(1,no_label):
        if label != label_2:
            print(label,label_2)
            Y= []
            X1 = []
            X2 = []
            for i in range(0,len(X1m)):
                if Y_train_1[i]==label:
                    Y.append(1)
                    X1.append(X1m[i])
                    X2.append(X2m[i])
                elif Y_train_1[i]==label_2:
                    Y.append(-1)
                    X1.append(X1m[i])
                    X2.append(X2m[i])
            Y=np.transpose(np.asmatrix(Y))
            
        
            l=np.count_nonzero(Y)
            u=len(Y)-l
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
            gamma_I2=10 #gamma_I2=0 for supervised SVM
            
            sigma1=0.05
            gamma_A1=100
            gamma_I1=10 #gamma_I1=0 for supervised SVM
            
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
              
            for i in range(0,l,1):
                J1[i,i]=1
                Yd1[i,i]=Y[i]
            for i in range(0,n,1):
                xi=X1[i]
                for j  in range(0,n,1):
                    xj=X1[j]
                    K1[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma1**2)))         
                    W1[i,j]=np.exp(-(LA.norm(xi-xj)))                              
            d=np.sum(W1,axis=1)
            for i in range(0,l+u,1):
                D1[i,i]=d[i]
            L1=D1-W1

            for i in range(0,l,1):
                J2[i,i]=1
                Yd2[i,i]=Y[i]
            for i in range(0,n,1):
                xi=X2[i]
                for j  in range(0,n,1):
                    xj=X2[j]
                    K2[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma2**2)))         
                    W2[i,j]=np.exp(-(LA.norm(xi-xj)))                              
            d=np.sum(W2,axis=1)
            for i in range(0,l+u,1):
                D2[i,i]=d[i]
            L2=D2-W2
            
            alpha_multi=0.5
            L=(1-alpha_multi)*L1+(alpha_multi)*L2
            
            Q2=Yd2*J2*K2*(LA.inv((2*gamma_A2*np.eye(n))+((2*gamma_I2/(n^2))*L*K2)))*np.transpose(J2)*Yd2
            Q1=Yd1*J1*K1*(LA.inv((2*gamma_A1*np.eye(n))+((2*gamma_I1/(n^2))*L*K1)))*np.transpose(J1)*Yd1
            
            
            y=(Y)
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
            
            
            
            alpha_final1[:,label-1,label_2-1] = np.transpose(alpha1)
            alpha_final2[:,label-1,label_2-1] = np.transpose(alpha2)
            
            for i in range(0,len(X_test_1)):
                f1=0
                for j in range(0,l):
                    Kx1=np.exp(-((X_test_1[i]-X1[j])*np.transpose(X_test_1[i]-X1[j]))/sigma1)
                    f1=f1 + (alpha_final1[j,label-1,label_2-1]*Kx1)
                    f_final1[i,label-1,label_2-1]= f1
            
            for i in range(0,len(X_test_2)):
                f2=0
                for j in range(0,l):
                    Kx2=np.exp(-((X_test_2[i]-X2[j])*np.transpose(X_test_2[i]-X2[j]))/sigma2)
                    f2=f2 + (alpha_final2[j,label-1,label_2-1]*Kx2)
                    f_final2[i,label-1,label_2-1]= f2
                    
for i in range(0,len(X_test_1)):
    f1_values = np.matrix.sum(np.asmatrix(f_final1[i,:,:]),1)  
    f2_values = np.matrix.sum(np.asmatrix(f_final2[i,:,:]),1) 
    lar1 = max(f1_values)
    lar2 = max(f2_values)
    if lar1>lar2:
        Y_predcted[i]=[p+1 for p, z in enumerate(f1_values) if z == lar1][0]
    else:
        Y_predcted[i]=[p+1 for p, z in enumerate(f2_values) if z == lar2][0]
    if Y_predcted[i]==Y_test_1[i]:
         n_crrct=n_crrct+1

Percnt_crrct=100*n_crrct/float(len(X_test_1))       
print(Percnt_crrct) 
            
