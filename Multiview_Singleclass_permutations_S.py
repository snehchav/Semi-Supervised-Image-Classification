import math 
import numpy as np
import numpy.matlib
import scipy.io as sio
from numpy import linalg as LA
import os

def grad(alpha,K,Y,L,J,l,u,gamma_I,gamma_A):
    A=J*K*alpha
    B=gamma_A*l*alpha
    C=(gamma_I/(l+u)**2)*L*K*alpha
    d=A+B+C-Y  
    return(d)


mat1 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fourth Feature\smaller\X_train.mat')
mat2 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fourth Feature\smaller\Y_train.mat')
mat3 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fourth Feature\smaller\X_test.mat')
mat4 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fourth Feature\smaller\Y_test.mat')
X1=np.asmatrix(mat1['X_final'])
Y1_trncomplte=np.asmatrix(mat2['Y_final'])
X1_test=np.asmatrix(mat3['X_test'])
Y1_tstcomplte=np.asmatrix(mat4['Y_test'])
Y1_test=np.transpose(np.asmatrix((-1)*np.ones(len(Y1_tstcomplte))))

X1=X1[0:1500,:]
Y1_trncomplte=Y1_trncomplte[0:1500,:]
Y1=np.transpose(np.asmatrix((-1)*np.ones(len(Y1_trncomplte))))

mat1 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\X_train.mat')
mat2 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\Y_train.mat')
mat3 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\X_test.mat')
mat4 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\Y_test.mat')
X2=np.asmatrix(mat1['X_final'])
Y2_trncomplte=np.asmatrix(mat2['Y_final'])
X2_test=np.asmatrix(mat3['X_test'])
Y2_tstcomplte=np.asmatrix(mat4['Y_test'])
Y2_test=np.transpose(np.asmatrix((-1)*np.ones(len(Y2_tstcomplte))))

X2=X2[0:1500,:]
Y2_trncomplte=Y2_trncomplte[0:1500,:]
Y2=np.transpose(np.asmatrix((-1)*np.ones(len(Y2_trncomplte))))

S1 = [1000000, 100000, 25000, 1e8, 500000 ]
S2 = [0.01]
G = [0.5]
Per = np.zeros((len(S1),len(S2),len(G)))
for s1 in range(0,len(S1)):
    for s2 in range(0,len(S2)):
        for g in range(0,len(G)):
            l=1500
            u=0
            #sigma1=float(1)
            #sigma2=float(10000)
            sigma1 = S1[s1]
            sigma2 = S2[s2]   
            L=np.matlib.zeros((l+u,l+u))
            L1=np.matlib.zeros((l+u,l+u))
            W1=np.matlib.zeros((l+u,l+u))
            D1=np.matlib.zeros((l+u,l+u))
            K1=np.matlib.zeros((l+u,l+u))
            L2=np.matlib.zeros((l+u,l+u))
            W2=np.matlib.zeros((l+u,l+u))
            D2=np.matlib.zeros((l+u,l+u))
            K2=np.matlib.zeros((l+u,l+u))
            f1=np.matlib.zeros((len(Y2_test),1))
            f2=np.matlib.zeros((len(Y2_test),1))
            Y_predcted=np.matlib.zeros((len(X1_test),1))
            
            label=3
            for i in range(0,len(Y1_trncomplte)):
                if Y1_trncomplte[i]==label:
                    Y1[i]=1
            for i in range(0,len(Y1_tstcomplte)):
                if Y1_tstcomplte[i]==label:
                    Y1_test[i]=1
            
            for i in range(0,len(Y2_trncomplte)):
                if Y2_trncomplte[i]==label:
                    Y2[i]=1
            for i in range(0,len(Y2_tstcomplte)):
                if Y2_tstcomplte[i]==label:
                    Y2_test[i]=1
            
            for i in range(0,l,1):
                xi=X1[i]
                for j  in range(0,l,1):
                    xj=X1[j]
                    K1[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/sigma1)
                    W1[i,j]=np.exp(-(LA.norm(xi-xj))**2)
            d1=np.sum(W1,axis=1)
            for i in range(0,l+u,1):
                D1[i,i]=d1[i]
            L1=D1-W1
            
            for i in range(0,l,1):
                xi=X2[i]
                for j  in range(0,l,1):
                    xj=X2[j]
                    K2[i,j]=np.exp(-((xi-xj)*np.transpose(xi-xj))/sigma2)
                    W2[i,j]=np.exp(-(LA.norm(xi-xj))**2)
            d2=np.sum(W2,axis=1)
            for i in range(0,l+u,1):
                D2[i,i]=d2[i]
            L2=D2-W2
            
            alpha_final=np.matlib.zeros((l+u,2))
            alpha_multi=.5
            L=(1-alpha_multi)*L1+(alpha_multi)*L2
            
            ## Load Data for view 1
            for q in range (0,2):
                if q==0:
                    X=X1
                    Y=Y1
                    K=K1
                else: 
                    X=X2
                    Y=Y2 
                    K=K2
            
                #Y[5000:10000]=0
                ## Intialize Variables
                n_crrct=0
                gamma_I=float(.05)
                #gamma_A=float(1.25)
                gamma_A = G[g]
                itrn=40000
                error=1e-8
                alpha=np.matlib.zeros((l+u,itrn))
                alpha[:,0]=np.asmatrix(np.random.randn(l+u,1))
                a=np.matlib.zeros((l+u,itrn))
                J=np.matlib.zeros((l+u,l+u))
                I=np.asmatrix(np.identity(l+u))
                for i in range(0,l,1):
                    J[i,i]=1
                ## Accelerated Gradient Decent
                for i in range(0,itrn-1):
                     a[:,i]=grad(alpha[:,i],K,Y,L,J,l,u,gamma_I,gamma_A)
                     alpha[:,i+1]=alpha[:,i]-(0.001)*a[:,i]
                     z1=alpha[:,i+1]-alpha[:,i]
                     z2=alpha[:,i]-alpha[:,i-1]
                     z1n=LA.norm(z1)
                     z2n=LA.norm(z2)
                     e=np.abs(z1n-z2n)
                     print(i)
                     print(e)
                     print(error)
                     if e<=error:
                         break
                fnl=i+1
                alpha_final[:,q]=alpha[:,fnl]     
                
            ## Testing Data for View 1
            for i in range(0,len(X1_test)):
                f=0
                for j in range(0,l+u):
                    Kx=np.exp(-((X1_test[i]-X1[j])*np.transpose(X1_test[i]-X1[j]))/sigma1)
                    f=f + (alpha_final[j,0]*Kx) 
                f1[i]=f    
                          
            ## Testing Data for View 2
            
            for i in range(0,len(X2_test)):
                f=0
                for j in range(0,l+u):
                    Kx=np.exp(-((X2_test[i]-X2[j])*np.transpose(X2_test[i]-X2[j]))/sigma2)
                    f=f + (alpha_final[j,1]*Kx) 
                f2[i]=f  
                
            for i in range(0,len(X2_test)):
                if abs(f1[i])>abs(f2[i]):
                    f=f1[i]
                else:
                    f=f2[i]
                if f<=0:    
                    Y_predcted[i]=-1
                else:
                    Y_predcted[i]=1
                if Y_predcted[i]==Y2_test[i]:
                    n_crrct=n_crrct+1             
            
            
            Percnt_crrct=100*n_crrct/float(len(X1_test))       
            print(Percnt_crrct) 
            Per[s1,s2,g]= Percnt_crrct
