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

mat1 = sio.loadmat('C:\Users\snehchav\Desktop\Machine Learning\Fourth Feature\smaller\X_train.mat')
mat2 = sio.loadmat('C:\Users\snehchav\Desktop\Machine Learning\Fourth Feature\smaller\Y_train.mat')
mat3 = sio.loadmat('C:\Users\snehchav\Desktop\Machine Learning\Fourth Feature\smaller\X_test.mat')
mat4 = sio.loadmat('C:\Users\snehchav\Desktop\Machine Learning\Fourth Feature\smaller\Y_test.mat')
X1m=np.asmatrix(mat1['X_final'])
Y1m_trncomplte=np.asmatrix(mat2['Y_final'])
X1m_test=np.asmatrix(mat3['X_test'])
Y1m_tstcomplte=np.asmatrix(mat4['Y_test'])

X1m=X1m[0:1500,:]
Y1m_trncomplte=Y1m_trncomplte[0:1500,:]


mat1 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\X_train.mat')
mat2 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\Y_train.mat')
mat3 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\X_test.mat')
mat4 = sio.loadmat('C:\Users\sandippk\Desktop\Machine Learning\Fifth Feature\smaller\Y_test.mat')
X2m=np.asmatrix(mat1['X_final'])
Y2m_trncomplte=np.asmatrix(mat2['Y_final'])
X2m_test=np.asmatrix(mat3['X_test'])
Y2m_tstcomplte=np.asmatrix(mat4['Y_test'])

X2m=X2m[0:1500,:]
Y2m_trncomplte=Y2m_trncomplte[0:1500,:]

no_label=5

S1 = [1000000, 100000, 25000, 1e8, 500000 ]
S2 = [0.01]
G = [0.5]
Per = np.zeros((len(S1),len(S2),len(G)))
for s1 in range(0,len(S1)):
    for s2 in range(0,len(S2)):
        for g in range(0,len(G)):
            f_final1=np.zeros((len(X1m_test),no_label-1,no_label-1))
            f_final2=np.zeros((len(X1m_test),no_label-1,no_label-1))
            
            ## Selecting two classes loop
            alpha_final1 = np.zeros((600,no_label,no_label))
            alpha_final2 = np.zeros((600,no_label,no_label))
            f_final=np.zeros((len(X1m_test),no_label-1,no_label-1))
            for label in range(1,no_label):
                for label_2 in range(1,no_label):
                    if label != label_2:
                        print(label,label_2)
                        Y= []
                        X1 = []
                        X2 = []
                        for i in range(0,len(X1m)):
                            if Y1m_trncomplte[i]==label:
                                Y.append(1)
                                X1.append(X1m[i])
                                X2.append(X2m[i])
                            elif Y1m_trncomplte[i]==label_2:
                                Y.append(-1)
                                X1.append(X1m[i])
                                X2.append(X2m[i])
                        Y=np.transpose(np.asmatrix(Y))
                        #Y[2000:4000]=0
                        ## Unlabeling 
                        #Y[500:6000]=0    
                        l=np.count_nonzero(Y)
                        u=len(X1)-l
                        sigma1 = S1[s1]
                        sigma2 = S2[s2]
                        #sigma1=float(1000000)
                        #sigma2=float(1)
                        L=np.matlib.zeros((l+u,l+u))
                        L1=np.matlib.zeros((l+u,l+u))
                        W1=np.matlib.zeros((l+u,l+u))
                        D1=np.matlib.zeros((l+u,l+u))
                        K1=np.matlib.zeros((l+u,l+u))
                        L2=np.matlib.zeros((l+u,l+u))
                        W2=np.matlib.zeros((l+u,l+u))
                        D2=np.matlib.zeros((l+u,l+u))
                        K2=np.matlib.zeros((l+u,l+u))
                        f1=np.matlib.zeros((len(Y),1))
                        f2=np.matlib.zeros((len(Y),1))
                        Y_predcted=np.matlib.zeros((len(X1m_test),1))
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
                                K=K1
                            else: 
                                X=X2
                                K=K2
                        
                            ## Intialize Variables
                            n_crrct=0
                            gamma_I = G[g]
                            #gamma_I=float(.5)
                            gamma_A=float(0.005)
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
                                 #print(i)
                                 print(e)
                                 #print(error)
                                 if e<=error:
                                     break
                            fnl=i+1   
                            if q==0:
                                alpha_final1[:,label-1,label_2-1] = np.reshape(alpha[:,fnl],600,0)
                                for i in range(0,len(X1m_test)):
                                    f1=0
                                    for j in range(0,l+u):
                                        Kx1=np.exp(-((X1m_test[i]-X[j])*np.transpose(X1m_test[i]-X[j]))/sigma1)
                                        f1=f1 + (alpha_final1[j,label-1,label_2-1]*Kx1)
                                    f_final1[i,label-1,label_2-1]= f1
                            else:
                                alpha_final2[:,label-1,label_2-1] = np.reshape(alpha[:,fnl],600,0)
                                for i in range(0,len(X1m_test)):
                                    f2=0
                                    for j in range(0,l+u):
                                        Kx2=np.exp(-((X2m_test[i]-X[j])*np.transpose(X2m_test[i]-X[j]))/sigma2)
                                        f2=f2 + (alpha_final2[j,label-1,label_2-1]*Kx2)
                                    f_final2[i,label-1,label_2-1]= f2
            
            for i in range(0,len(X1m_test)):
                f1_values = np.matrix.sum(np.asmatrix(f_final1[i,:,:]),1)  
                f2_values = np.matrix.sum(np.asmatrix(f_final2[i,:,:]),1) 
                lar1 = max(f1_values)
                lar2 = max(f2_values)
                if lar1>lar2:
                    Y_predcted[i]=[p+1 for p, z in enumerate(f1_values) if z == lar1][0]
                else:
                    Y_predcted[i]=[p+1 for p, z in enumerate(f2_values) if z == lar2][0]
                if Y_predcted[i]==Y1m_tstcomplte[i]:
                     n_crrct=n_crrct+1
            
            Percnt_crrct=100*n_crrct/float(len(X1m_test))       
            print(Percnt_crrct) 
            Per[s1,s2,g]= Percnt_crrct
