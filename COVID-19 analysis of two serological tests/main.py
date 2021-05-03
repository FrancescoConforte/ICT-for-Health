import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
def findROC(x,y):# 
    """ findROC(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""
    
    x0=x[y==0] #  test values for healthy patients
    x1=x[y==1] #  test values for ill patients
    xss=np.sort(x) #  sort test values to get all the possible  thresholds
    xs=np.unique(xss) #  remove repetitions
    if xs[0]>0:
        xs=np.insert(xs,0,0) # add 0 as first element 
    Np=np.sum(y==1) #  number of ill patients
    Nn=np.sum(y==0) #  number of healthy patients
    data=np.zeros((len(xs),3),dtype=float)
    
    for i,thresh in enumerate(xs):
        n1=np.sum(x1>thresh)
        sens=n1/Np
        n2=np.sum(x0<thresh)
        spec=n2/Nn
        FA=1-spec
        data[i,0]=thresh
        data[i,1]=FA #false positive or false alarm
        data[i,2]=sens #true positive or sensitivity
    return data

    # if x> thresh -> test is positive
    # if x < thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of positives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y


plt.close('all')
xx=pd.read_csv("./data/covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= neg., 1 = unclear, 2=pos.
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
# remove uncertain swab tests
ii=np.argwhere(swab==1).flatten()
swab=np.delete(swab,ii)
swab=swab//2
Test1=np.delete(Test1,ii)
Test2=np.delete(Test2,ii)    

     #======================Test 1======================
data_Test1=findROC(Test1,swab)

#Area Under the Curve (AUC) for Test 1
FPR_Test_1 = data_Test1[:,1] #False Positive Rate or False Alarm
TPR_Test_1 = data_Test1[:,2] #True Positive Rate or Sensitivity
AUC_Test_1 = metrics.auc(FPR_Test_1,TPR_Test_1) #AUC from sklearn   
AUC_Test_1_new = -1 * np.trapz(TPR_Test_1, FPR_Test_1) #AUC with integral using Trapezoidal Rule

#Plot ROC 
plt.figure()
plt.plot(data_Test1[:,1],data_Test1[:,2],label='Test1')
plt.axis([0,1,0,1])
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend(loc='center right')
plt.title('ROC for Test1')

#Plot Specificity and Sensitivity vs Thresholds
plt.figure()
plt.plot(data_Test1[:,0],1-data_Test1[:,1],label='spec.')#specificity
plt.plot(data_Test1[:,0],data_Test1[:,2],label='sens.')#sensitivity
plt.grid()
plt.xlabel('Threshold')
plt.title('Test1')
plt.legend()

prev=0.025
spec=1-data_Test1[:,1]
sens=data_Test1[:,2]
PDgivenTp=sens*prev/(sens*prev+(1-spec)*(1-prev)+1e-8)
PHgivenTn=spec*(1-prev)/(spec*(1-prev)+(1-sens)*prev+1e-8)
PDgivenTn=1-PHgivenTn

soglia1 = np.array([data_Test1[:,0],sens,spec,PDgivenTn,PDgivenTp]).T
soglia1= pd.DataFrame(soglia1,columns=['Threshold','sensitivity','specificity','P(D|Tn)','P(D|Tp)'])

#Probabilities P(D|Tp) and P(D|Tn) vs thresholds
plt.figure()
plt.semilogy(data_Test1[:,0],PDgivenTp,label='P(D|Tp)')
plt.semilogy(data_Test1[:,0],PDgivenTn,label='P(D|Tn)')
plt.legend()
plt.grid()
plt.xlabel('Threshold')

plt.figure()
plt.semilogy(data_Test1[:,0],PDgivenTp,label='P(D|Tp)')
plt.semilogy(data_Test1[:,0],PDgivenTn,label='P(D|Tn)')
plt.plot(7.5,PDgivenTn[121],marker='o',c='r',label='Chosen Threshold=7.5')
plt.plot(7.5,PDgivenTp[121],marker='o',c='r')
plt.legend(loc='best')
plt.grid()
plt.xlim(right=101,left=-7)
plt.ylim(top=10**-(1/3))
plt.xlabel('Threshold')

     #======================Test 2======================
data_Test2=findROC(Test2,swab) 

#Area Under the Curve (AUC) for Test 2
FPR_Test_2 = data_Test2[:,1] #False Positive Rate or False Alarm
TPR_Test_2 = data_Test2[:,2] #True Positive Rate or Sensitivity
AUC_Test_2 = metrics.auc(FPR_Test_2,TPR_Test_2) #AUC from sklearn   
AUC_Test_2_new = -1 * np.trapz(TPR_Test_2, FPR_Test_2) #AUC with integral using Trapezoidal Rule

#Plot ROC
plt.figure()
plt.plot(data_Test2[:,1],data_Test2[:,2],label='Test2')
plt.axis([0,1,0,1])
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend(loc='center right')
plt.title('ROC for Test2')

#Plot Specificity and Sensitivity vs Thresholds  
plt.figure()
plt.plot(data_Test2[:,0],1-data_Test2[:,1],label='spec.')
plt.plot(data_Test2[:,0],data_Test2[:,2],label='sens.')
plt.grid()
plt.xlabel('Threshold')
plt.title('Test2')
plt.legend()

prev=0.025
spec=1-data_Test2[:,1]
sens=data_Test2[:,2]
PDgivenTp=sens*prev/(sens*prev+(1-spec)*(1-prev)+1e-8)
PHgivenTn=spec*(1-prev)/(spec*(1-prev)+(1-sens)*prev+1e-8)
PDgivenTn=1-PHgivenTn

soglia2 = np.array([data_Test2[:,0],sens,spec,PDgivenTn,PDgivenTp]).T
soglia2= pd.DataFrame(soglia2,columns=['Threshold','sensitivity','specificity','P(D|Tn)','P(D|Tp)'])

#Probabilities P(D|Tp) and P(D|Tn) vs thresholds
plt.figure()
plt.semilogy(data_Test2[:,0],PDgivenTp,label='P(D|Tp)')
plt.semilogy(data_Test2[:,0],PDgivenTn,label='P(D|Tn)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('Threshold')

plt.figure()
plt.semilogy(data_Test2[:,0],PDgivenTp,label='P(D|Tp)')
plt.semilogy(data_Test2[:,0],PDgivenTn,label='P(D|Tn)')
plt.plot(0.3,PDgivenTn[26],marker='o',c='r',label='Chosen Threshold=0.3')
plt.plot(0.3,PDgivenTp[26],marker='o',c='r')
plt.legend(loc='best')
plt.grid()
plt.xlim(right=2,left=-0.2)
plt.ylim(top=10**-(1/3))
plt.xlabel('Threshold')

#Application of Tests on dataset
positives_Test1 = len(np.argwhere(Test1>=7.5).flatten()) #162
positives_Test2 = len(np.argwhere(Test2>=0.3).flatten()) #143
positives_swab = len(np.argwhere(swab==1).flatten()) #71

