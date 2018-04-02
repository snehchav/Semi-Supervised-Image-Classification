# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:24:28 2017

@author: Team 6
"""

import math 
import numpy as np
import numpy.matlib
import scipy.io as sio
from numpy import linalg as LA

import cvxopt
from cvxopt import solvers
from cvxopt import matrix

## Load Data
mat1 = sio.loadmat('X_train_2.mat')  # LOAD PIXEL DATASET
mat2 = sio.loadmat('Y_train_2.mat')
mat3 = sio.loadmat('X_test_2.mat')
mat4 = sio.loadmat('Y_test_2.mat')
X1=np.asmatrix(mat1['X_final'])
Y_trncomplte=np.asmatrix(mat2['Y_final'])
X_test=np.asmatrix(mat3['X_test'])
Y_tstcomplte=np.asmatrix(mat4['Y_test'])

Y_predcted=np.matlib.zeros((len(X_test),1))
no_label = 6
n_crrct=0
Y_trncomplte=np.int32(Y_trncomplte)
Y_tstcomplte=np.int32(Y_tstcomplte)

alpha_final = np.zeros((1200,no_label-1,no_label-1))
f_final=np.zeros((len(X_test),no_label-1,no_label-1))

sigma=200                          #Defining Parameters
gamma_A=5
gamma_I=5                            #Supervised gamma_I=0


for label in range(1,no_label):
    for label_2 in range(1,no_label):
        if label != label_2:
            print(label)
            print(label_2)
            Y= []
            X = []
            for i in range(0,3000):
                if Y_trncomplte[i]==label:
                    Y.append(1)
                    X.append(X1[i])
                elif Y_trncomplte[i]==label_2:
                    Y.append(-1)
                    X.append(X1[i])
            Y = np.transpose(np.matrix(Y))

            l=np.count_nonzero(Y)                   #Defining Data
            u=len(Y)-l
            n=l+u
            alpha=np.matlib.zeros((l+u,1))
            beta=np.matlib.zeros((l,1))
            K=np.matlib.zeros((l+u,l+u))
            Kx=np.matlib.zeros((len(X_test),l+u))
            J=np.matlib.zeros((l,l+u))
            L=np.matlib.zeros((l+u,l+u))
            W=np.matlib.zeros((l+u,l+u))
            D=np.matlib.zeros((l+u,l+u))
            Q=np.matlib.zeros((l,l))
            Yd=np.matlib.zeros((l,l))

            for i in range(0,l,1):
                J[i,i]=1
                Yd[i,i]=Y[i]
            for i in range(0,n,1):
                xi=X[i]
                for j  in range(0,n,1):
                    xj=X[j]
                    K[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/(2*(sigma**2)))         
                    W[i,j]=np.exp(-(LA.norm(xi-xj))**2)                              
            d=np.sum(W,axis=1)
            for i in range(0,n,1):
                D[i,i]=d[i]
            L=D-W
            Q=Yd*J*K*(LA.inv((2*gamma_A*np.eye(n))+((2*gamma_I/(n^2))*L*K)))*np.transpose(J)*Yd
        

            y=(Y)                                                #Implementing SVM using CVXOPT 
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
            alpha_final[:,label-1,label_2-1]=np.transpose(alpha)
        
            for i in range(0,len(X_test)):
                f=0
                for j in range(0,l):
                    Kx=np.exp(-((X_test[i]-X[j])*np.transpose(X_test[i]-X[j]))/(2*(sigma**2)))
                    f=f + (alpha_final[j,label-1,label_2-1]*Kx)
                    f_final[i,label-1,label_2-1]= f
                
for i in range(0,len(X_test)):
    f_values = np.matrix.sum(np.asmatrix(f_final[i,:,:]),1)    
    lar = max(f_values)
    Y_predcted[i]=[p+1 for p, z in enumerate(f_values) if z == lar][0]
    if Y_predcted[i]==Y_tstcomplte[i]:
         n_crrct=n_crrct+1
         
Percnt_crrct=100*n_crrct/float(len(X_test))       
print(Percnt_crrct)     
