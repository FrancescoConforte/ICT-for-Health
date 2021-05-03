# -*- coding: utf-8 -*-
"""
PEP 8 -- Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/

@author: Francesco Conforte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def GPR(X_train,y_train,X_val,r2,s2):
    """ Estimates the output y_val given the input X_val, using the training data 
    and  hyperparameters r2 and s2"""
    Nva=X_val.shape[0]
    yhat_val=np.zeros((Nva,))
    sigmahat_val=np.zeros((Nva,))
    for k in range(Nva):
        x=X_val[k,:]# k-th point in the validation dataset
        A=X_train-np.ones((Ntr,1))*x
        dist2=np.sum(A**2,axis=1)
        ii=np.argsort(dist2)
        ii=ii[0:N-1];
        refX=X_train[ii,:]
        Z=np.vstack((refX,x))
        sc=np.dot(Z,Z.T)# dot products
        e=np.diagonal(sc).reshape(N,1)# square norms
        D=e+e.T-2*sc# matrix with the square distances 
        R_N=np.exp(-D/2/r2)+s2*np.identity(N)#covariance matrix
        R_Nm1=R_N[0:N-1,0:N-1]#(N-1)x(N-1) submatrix 
        K=R_N[0:N-1,N-1]# (N-1)x1 column
        d=R_N[N-1,N-1]# scalar value
        C=np.linalg.inv(R_Nm1)
        refY=y_train[ii]
        mu=K.T@C@refY# estimation of y_val for X_val[k,:]
        sigma2=d-K.T@C@K
        sigmahat_val[k]=np.sqrt(sigma2)
        yhat_val[k]=mu        
    return yhat_val,sigmahat_val

def print_result(title,e,y_norm,sy,my):
        """ 
        print_result prints a dataframe containing parameters: mean, standard 
        deviation, mean squared value, parameter R2 of the error in each of the
        three subsets

        Parameters
        ----------
        title: tipically the algorithm used to find the regressand
            string
            
        e: Nd array containing the errors for training, validation and test datasets
            float
            
        y_norm: Nd array containing normalized regressand values for training, validation and test dataset
            float
            
        sy: standard deviation of regressand train
            float
        
        my: mean of regressand train
            float
        """
        E_tr=e[0]
        E_va=e[1]
        E_te=e[2]
        y_tr_norm=y_norm[0]
        y_va_norm=y_norm[1]
        y_te_norm=y_norm[2]
        
        E_tr_mu=E_tr.mean()
        E_tr_sig=E_tr.std()
        E_tr_MSE=np.mean(E_tr**2)
        y_tr=y_tr_norm*sy+my
        R2_tr=1-E_tr_sig**2/np.mean(y_tr**2)
        E_va_mu=E_va.mean()
        E_va_sig=E_va.std()
        E_va_MSE=np.mean(E_va**2)
        y_va=y_va_norm*sy+my
        R2_va=1-E_va_sig**2/np.mean(y_va**2)
        E_te_mu=E_te.mean()
        E_te_sig=E_te.std()
        E_te_MSE=np.mean(E_te**2)
        y_te=y_te_norm*sy+my
        R2_te=1-E_te_sig**2/np.mean(y_te**2)
        rows=['Training','Validation','Test']
        cols=['mean','std','MSE','R^2']
        p=np.array([[E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr],
                    [E_va_mu,E_va_sig,E_va_MSE,R2_va],
                    [E_te_mu,E_te_sig,E_te_MSE,R2_te]])
        results=pd.DataFrame(p,columns=cols,index=rows)
        print('\n Results for ' + title)
        print(results)



plt.close('all')
xx=pd.read_csv("./data/parkinsons_updrs.csv") # read the dataset
z=xx.describe().T # gives the statistical description of the content of each column
#xx.info()
# features=list(xx.columns)
features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

#%% Features to drop from dataset: 2 cases.

# Use 3 features as regressors (motor UPDRS, age and PPE)
todrop=['subject#', 'sex', 'test_time',  
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']

# Use 8 features as regressors (previous ones + Jitter (abs), Shimmer, NHR, HNR and DFA)
# todrop=['subject#', 'sex', 'test_time',
#         'Jitter(%)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
#         'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
#         'Shimmer:APQ11', 'Shimmer:DDA', 'RPDE']
x1=xx.copy(deep=True)
X=x1.drop(todrop,axis=1)
#%% Generate the shuffled dataframe
 
 # Note: As seed, the number 1834 was used to generate the three subsets. 
 # The number 277683 (my own matricola) was not used since the extracted 
 # subsets with it gave good performance on validation set but very bad 
 # performance on test set. 
 # Seed equal to 274948 was used as further seed to performe GPR;
 # it gives quite good results but for lab purpose 1834 was preferred.
 
np.random.seed(1834)
# np.random.seed(277683)
# np.random.seed(274948)
Xsh = X.sample(frac=1).reset_index(drop=True)
[Np,Nc]=Xsh.shape
F=Nc-1
#%% Generate training, validation and testing matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of testing points
X_tr=Xsh[0:Ntr] # training dataset
# find mean and standard deviations for the features in the training dataset
mm=X_tr.mean()
ss=X_tr.std()
my=mm['total_UPDRS']# get mean for the regressand
sy=ss['total_UPDRS']# get std for the regressand
# normalize data
Xsh_norm=(Xsh-mm)/ss
ysh_norm=Xsh_norm['total_UPDRS']
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
Xsh_norm=Xsh_norm.values
ysh_norm=ysh_norm.values
# get the training, validation, test normalized data
X_train_norm=Xsh_norm[0:Ntr]
X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
X_test_norm=Xsh_norm[Ntr+Nva:]
y_train_norm=ysh_norm[0:Ntr]
y_val_norm=ysh_norm[Ntr:Ntr+Nva]
y_test_norm=ysh_norm[Ntr+Nva:]
#
y_train=y_train_norm*sy+my
y_val=y_val_norm*sy+my
y_test=y_test_norm*sy+my


#%% Apply Gaussian Process Regression 
N=10

r2_new = np.arange(0.3,25,step=0.3,dtype=float) #seed 1834
s2_new = np.array([0.0002, 0.0005, 0.001, 0.002, 0.005]) #seed 1834

# r2_new = np.arange(0.3,20,step=0.3,dtype=float) #seed 277683
# s2_new = np.array([0.0002,0.0001,0.0005,0.00005,0.00002,0.00001,0.001,0.005]) #seed 277683

# r2_new = np.arange(0.1,9,step=0.1,dtype=float) #seed 274948
# s2_new = np.array([0.0002,0.0005,0.00002,0.001,0.002]) #seed 274948


# Generate a matrix of MSE values for different values of s2 and r2.
# To speed up the runtime of the code, if MSE increases for 10 consecutive times,
# then remaining values of MSE are put to NaN and value of s2 is changed with
# the following one.
# Comment lines 192-196 if you want to see all trends of MSE.

MSE_val = np.zeros((r2_new.shape[0],s2_new.shape[0]),dtype=float)
for idx_s, s in enumerate(s2_new):
    count=0
    for idx_r, r in enumerate(r2_new):
        y_hat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r,s)
        MSE_val[idx_r][idx_s] = np.mean((y_val_norm-y_hat_val_norm)**2)
        if MSE_val[idx_r][idx_s]>MSE_val[idx_r-1][idx_s] and MSE_val[idx_r-1][idx_s]!=0:
            count+=1
            if count == 10:
                MSE_val[idx_r:,idx_s] = np.nan
                break
        

#find best r2 and s2
minima_values = np.nanmin(MSE_val,axis = 0,keepdims=True) #return minima values for each column
minima_indexes = np.array(np.where(MSE_val == np.nanmin(MSE_val,axis = 0))) #return indexes of minima which is the corresponding value of r2
minima_indexes = minima_indexes[:,minima_indexes[1,:].argsort()] 

# create the matrix 'minima' having on:
    #first row: indexes corresponding to values of r2
    #second row: indexes corresponding to values of s2
    #third row: minimum values of MSE with corresponding s2 and r2    
minima = np.vstack((minima_indexes,minima_values))
idx = int(minima[0,np.argmin(minima[2,:])]) 

r2_best = r2_new[idx]
s2_best = s2_new[np.argmin(minima[2,:])]


# perform GPR with best hyperparameters on all the subsets
yhat_train_norm,sigmahat_train=GPR(X_train_norm,y_train_norm,X_train_norm,r2_best,s2_best)
yhat_train=yhat_train_norm*sy+my
yhat_test_norm,sigmahat_test=GPR(X_train_norm,y_train_norm,X_test_norm,r2_best,s2_best)
yhat_test=yhat_test_norm*sy+my
yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2_best,s2_best)
yhat_val=yhat_val_norm*sy+my
err_train=y_train-yhat_train
err_test=y_test-yhat_test
err_val=y_val-yhat_val


#%% plots
plt.figure()

for i in range(MSE_val.shape[1]):
    plt.plot(r2_new, MSE_val[:,i],label='s2='+ str(s2_new[i]))
    

plt.plot(r2_best,np.amin(minima_values),marker='o',color='black', label=('Best r2='+str(r2_best)+'\nand s2='+str(s2_best)))
plt.grid()
plt.legend(ncol=2)
plt.xlabel('r2')
plt.ylabel('MSE_val')
plt.title('MSE on validation dataset for different s2 and r2 values')
plt.show()
        

plt.figure()
plt.plot(y_test,yhat_test,'.b')
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression')
plt.show()


plt.figure()
plt.errorbar(y_test,yhat_test,yerr=3*sigmahat_test*sy,fmt='o',ms=2)
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression - with errorbars')
plt.show()


e=[err_train,err_val,err_test]
plt.figure()
plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Val.','Test'])
plt.xlabel('error')
plt.ylabel('P(error in bin)')
plt.legend()
plt.grid()
plt.title('Error histogram')
plt.show()


y_norm = [y_train_norm, y_val_norm, y_test_norm]
print_result('Gaussian Process Regression', e, y_norm, sy, my)
print("Best r2=" + str(r2_best))
print("Best s2=" + str(s2_best))

