# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:45:29 2020

@author: Francesco Conforte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sub.minimization as mymin

plt.close('all')
x=pd.read_csv("data/parkinsons_updrs.csv") # read the dataset; xx is a dataframe
x.describe().T # gives the statistical description of the content of each column
x.info()
features=list(x.columns)
print(features)
#features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
#       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
#       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
#       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
X=x.drop(['subject#','test_time'],axis=1)# drop unwanted features
Np,Nc=X.shape# Np = number of rows/ptients Nf=number of features+1 (total UPDRS is included)
features=list(X.columns)
#%% correlation
Xnorm=(X-X.mean())/X.std()# normalized data
c=Xnorm.cov()# xx.cov() gives the wrong result

plt.figure(figsize=(10,10))
plt.matshow(np.abs(c.values),fignum=0)
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)    
plt.colorbar()
plt.title('Covariance matrix of the features',pad=70)

plt.figure()
c.motor_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)#, **kwargs) 
plt.title('corr coeff among motor UPDRS and the other features')
plt.gcf().subplots_adjust(bottom=0.3)
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation=90)#, **kwargs) 
plt.title('corr coeff among total UPDRS and the other features')
plt.gcf().subplots_adjust(bottom=0.3)

#%% Generate the shuffled data
np.random.seed(1) # set the seed for random shuffling
indexsh=np.arange(Np)
np.random.shuffle(indexsh)
Xsh=X.copy(deep=True)
Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False)
Xsh=Xsh.sort_index(axis=0)
#%% Generate training, validation and test matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series)
ss=X_tr.std()# standard deviation (series)
my=mm['total_UPDRS']# mean of motor UPDRS
sy=ss['total_UPDRS']# st.dev of motor UPDRS
#%% Normalize the three subsets
Xsh_norm=(Xsh-mm)/ss# normalized data
ysh_norm=Xsh_norm['total_UPDRS']# regressand only
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)# regressors only

X_tr_norm=Xsh_norm[0:Ntr] #training regressors
X_va_norm=Xsh_norm[Ntr:Ntr+Nva] #validation regressors
X_te_norm=Xsh_norm[Ntr+Nva:] #test regressors
y_tr_norm=ysh_norm[0:Ntr] #training regressand
y_va_norm=ysh_norm[Ntr:Ntr+Nva] #validation regressand
y_te_norm=ysh_norm[Ntr+Nva:] #test regressand

y_norm=[y_tr_norm,y_va_norm,y_te_norm] 
out=np.empty((9,))

#%% Linear Least Squares
lls = mymin.SolveLLS(y_tr_norm.values, X_tr_norm.values)
w_hat = lls.run()
#lls.plot_w_hat(X_tr_norm, title='Optimized weights - Linear Least Squares')

#w_hat=lls.sol
E_tr=(y_tr_norm-X_tr_norm@w_hat)*sy# training
E_va=(y_va_norm-X_va_norm@w_hat)*sy# validation
E_te=(y_te_norm-X_te_norm@w_hat)*sy# test
e=[E_tr,E_va,E_te]

y_hat_te_norm=X_te_norm@w_hat
MSE_norm_lls=np.mean((y_hat_te_norm-y_te_norm)**2)
MSE_lls=sy**2*MSE_norm_lls

y_hat_te=y_hat_te_norm*sy+my
y_te=y_te_norm*sy+my

lls.plot_w_hat(X_tr_norm, title='Optimized weights - Linear Least Squares')
lls.plot_hist('LLS',e)
lls.plot_y_hat_vs_y('Linear Least Squares: test',y_te,y_hat_te)
lls.print_result('Linear Least Squares',e,y_norm,sy,my)


#%% Stochastic Gradient with Adam
swa = mymin.SolveStochWithADAM(y_tr_norm.values, X_tr_norm.values,y_va_norm.values,X_va_norm.values)
w_hat = swa.run(gamma=1e-4,Nit=300000,sy=sy)
#swa.plot_w_hat(X_tr_norm, title='Optimized weights - Stochastic Gradient Algorithm with ADAM')

#w_hat=swa.sol
E_tr=(y_tr_norm-X_tr_norm@w_hat)*sy# training
E_va=(y_va_norm-X_va_norm@w_hat)*sy# validation
E_te=(y_te_norm-X_te_norm@w_hat)*sy# test
e=[E_tr,E_va,E_te]

y_hat_te_norm=X_te_norm@w_hat
MSE_norm_swa=np.mean((y_hat_te_norm-y_te_norm)**2)
MSE_swa=sy**2*MSE_norm_swa

y_te=y_te_norm*sy+my
y_hat_te=y_hat_te_norm*sy+my

swa.plot_w_hat(X_tr_norm, title='Optimized weights - Stochastic Gradient Algorithm with ADAM')
swa.plot_hist('Stochastic Gradient with ADAM',e)
swa.plot_err('Stochastic Gradient with ADAM: Mean Squared Error',1,0)
swa.plot_y_hat_vs_y('Stochastic Gradient with ADAM: test',y_te,y_hat_te)
swa.print_result('Stochastic Gradient with ADAM',e,y_norm,sy,my)

#%% Ridge Regression
rr = mymin.SolveRidge(y_tr_norm.values, X_tr_norm.values,y_va_norm.values,X_va_norm.values)
possible_lambdas = np.arange(101)
errors_tr = np.zeros((len(possible_lambdas),2),dtype=float)
errors_val = np.zeros((len(possible_lambdas),2),dtype=float)

for i in possible_lambdas:
    w_lambda = rr.run(i)
    #w_lambda = rr.sol
    errors_tr[i,0]=i
    errors_tr[i,1]=sy**2*np.mean((X_tr_norm.values@w_lambda-y_tr_norm.values)**2)
    errors_val[i,0]=i
    errors_val[i,1]=sy**2*np.mean((X_va_norm.values@w_lambda-y_va_norm.values)**2)
 
best = np.argmin(errors_val[:,1])
rr.plot_MSEvsLambda(errors_tr, errors_val, best)
w_hat = rr.run(best)

E_tr=(y_tr_norm-X_tr_norm@w_hat)*sy# training
E_va=(y_va_norm-X_va_norm@w_hat)*sy# validation
E_te=(y_te_norm-X_te_norm@w_hat)*sy# test
e=[E_tr,E_va,E_te]

y_hat_te_norm=X_te_norm@w_hat
MSE_norm_rr=np.mean((y_hat_te_norm-y_te_norm)**2)
MSE_rr=sy**2*MSE_norm_rr

y_hat_te=y_hat_te_norm*sy+my
y_te=y_te_norm*sy+my

rr.plot_w_hat(X_tr_norm, title='Optimized weights - Ridge Regression')
rr.plot_hist('Ridge Regression',e)
rr.plot_y_hat_vs_y('Ridge Regression: test',y_te,y_hat_te)
rr.print_result('Ridge Regression',e,y_norm,sy,my)







