import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix

plt.close('all')

# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
# import the dataframe:
#xx=pd.read_csv("./Chronic_Kidney_Disease/chronic_kidney_disease.arff",sep=',',
#               skiprows=29,names=feat_names, 
#               header=None,na_values=['?','\t?'],
#               warn_bad_lines=True)
xx=pd.read_csv("./data/Chronic_Kidney_Disease/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
key_list=["normal","abnormal","present","notpresent","yes",
"no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
xx=xx.replace(key_list,key_val)
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

#%% manage the missing data through regression
print(xx.info())
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19) 
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('max number of missing values in the reduced dataset: ',n.max())
print('number of points in the reduced dataset: ',len(n))
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain.iloc[:,k]
        val=val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')

#normalize the training dataset
mm=Xtrain.mean(axis=0)
ss=Xtrain.std(axis=0)
Xtrain_norm=(Xtrain-mm)/ss
#normalize the entire dataset using the coeffs found for the training dataset
X_norm=(x-mm)/ss
Np,Nf=X_norm.shape


##Substitution of missing values with linear regression
x_new=x.copy() # make a copy of the dataframe
miss_values=x.isnull().sum(axis=1) # find the number of missing values per row
nn=np.sort(miss_values.unique())[1:]
for i in nn:
    rows=(miss_values==i).values # rows with i missing value
    rows=np.argwhere(rows).flatten() # indexes of rows with one missing value
    for kk in rows:
        xrow=X_norm.iloc[kk] # get the row
        mask=xrow.isna() # with the column with nan in the row
        index=np.array(np.where(mask.values==True)).flatten() # indexes of this column
        for m in range(index.shape[0]):
            Data_tr_norm=Xtrain_norm.loc[:,~mask] # training data: regressors
            y_tr_norm=Xtrain_norm.values[:,index[m]] #training data: regressand
            # find the vector with weights
            w=(np.linalg.inv(Data_tr_norm.T@Data_tr_norm))@(Data_tr_norm.T@y_tr_norm)
            # find the regressed value for the row
            yrow_norm=xrow[~mask].values@w # normalized
            yrow=yrow_norm*ss.values[index[m]]+mm.values[index[m]]# denormalized
            if feat_cat[index[m]]=='cat': # if the missing value is of a categorical feat.
                val=alphabets[index[m]] # find the possible values
                d=(val-yrow)**2 # sq. dist. between regressed and alphabet values
                yrow=val[d.argmin()] # alphabet value closer to the regressed value
                x_new.iloc[kk,index[m]]=yrow # remove nan and put the regressed value
            else:
                x_new.iloc[kk,index[m]]=np.round(yrow) # remove nan and put the regressed value
    X_norm=(x_new-mm)/ss
    Np,Nf=X_norm.shape
print(x_new.info())

#Substitution of missing values with median computed on training
x_new_median = xx.fillna(Xtrain.median())
print(x_new_median.info())

##------------------ Decision tree -------------------
##
target_names = ['notckd','ckd']
# Let us use only the complete data (no missing values)
N_measurements = 200
acc_scores=np.zeros((N_measurements,),dtype=float)
target = Xtrain.loc[:,'classk']
inform = Xtrain.drop('classk', axis=1)
sensitivity=np.zeros((N_measurements,),dtype=float)
specificity=np.zeros((N_measurements,),dtype=float)
# PPV=np.zeros((N_measurements,),dtype=float)
# NPV=np.zeros((N_measurements,),dtype=float)

for i in range(N_measurements):   
    clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=i)
    clfXtrain = clfXtrain.fit(inform,target)
    test_pred = clfXtrain.predict(x_new_median.drop('classk', axis=1))
    acc_scores[i]=accuracy_score(x_new_median.loc[:,'classk'], test_pred)
    tn, fp, fn, tp = confusion_matrix(x_new_median['classk'].values,test_pred).ravel()
    sensitivity[i]=tp/(tp+fn) #Sensitivity 
    specificity[i]=tn/(tn+fp) #Specificity
    # PPV[i]=tp/(tp+fp)
    # NPV[i]=tn/(fn+tn)
print('accuracy mean value = ', np.round(acc_scores.mean(),decimals=3))
print('accuracy max value = ', acc_scores.max())
print('accuracy min value = ', acc_scores.min())
print('sensitivity mean value = ',np.round(sensitivity.mean(),decimals=3))
print('sensitivity max value = ', sensitivity.max())
print('sensitivity min value = ', sensitivity.min())
print('specificity mean value = ',specificity.mean())
print('specificity max value = ', specificity.max())
print('specificity min value = ', specificity.min())
# print('PPV mean value = ', np.round(PPV.mean(),decimals=3))
# print('PPV max value = ', PPV.max())
# print('PPV min value = ', PPV.min())
# print('NPV mean value = ', np.round(NPV.mean(),decimals=3))
# print('NPV max value = ', NPV.max())
# print('NPV min value = ', NPV.min())
# Let us use the dataset with regressed missing values
target = x_new.loc[:,'classk']
inform = x_new.drop('classk', axis=1)
clfX = tree.DecisionTreeClassifier(criterion='entropy')
clfX = clfX.fit(inform,target)

# Let us use the dataset with MEDIAN values
target = x_new_median.loc[:,'classk']
inform = x_new_median.drop('classk', axis=1)
clfX_median = tree.DecisionTreeClassifier(criterion='entropy')
clfX_median = clfX_median.fit(inform,target)

##%% export to graghviz to draw a grahp
dot_data = tree.export_graphviz(clfXtrain,out_file='Tree_Train.dot',feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True)
dot_data_regr = tree.export_graphviz(clfX,out_file='Tree_w_regr.dot',feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True)
dot_data_med = tree.export_graphviz(clfX_median,out_file='Tree_median.dot',feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True)

## command: 
## $ dot -Tpng Tree.dot -o Tree.png


# ## Statistical results of tree obtained on the training tree and tested 
# ## over the regressed one
# target_names = ['notckd','ckd']
# # Let us use only the complete data (no missing values)
# N_measurements = 200
# acc_scores=np.zeros((N_measurements,),dtype=float)
# target = Xtrain.loc[:,'classk']
# inform = Xtrain.drop('classk', axis=1)
# sensitivity=np.zeros((N_measurements,),dtype=float)
# specificity=np.zeros((N_measurements,),dtype=float)
# F1_score=np.zeros((N_measurements,),dtype=float)
# for i in range(N_measurements):   
#     clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=i)
#     clfXtrain = clfXtrain.fit(inform,target)
#     test_pred = clfXtrain.predict(x_new.drop('classk', axis=1))
#     acc_scores[i]=accuracy_score(x_new.loc[:,'classk'], test_pred)
#     tn, fp, fn, tp = confusion_matrix(x_new['classk'].values,test_pred).ravel()
#     sensitivity[i]=tp/(tp+fn) #Sensitivity 
#     specificity[i]=tn/(tn+fp) #Specificity

# print('accuracy mean value = ', np.round(acc_scores.mean(),decimals=3))
# print('accuracy max value = ', acc_scores.max())
# print('accuracy min value = ', acc_scores.min())
# print('sensitivity mean value = ',np.round(sensitivity.mean(),decimals=3))
# print('sensitivity max value = ', sensitivity.max())
# print('sensitivity min value = ', sensitivity.min())
# print('specificity mean value = ',specificity.mean())
# print('specificity max value = ', specificity.max())
# print('specificity min value = ', specificity.min())