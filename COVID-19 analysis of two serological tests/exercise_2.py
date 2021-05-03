# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:47:36 2020

@author: Francesco Conforte
"""
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

def classe(X1,X2):
    '''
    Method to compute the class of the point

    Parameters
    ----------
    X1 : float
        X1 point (random variable).
    X2 : float
        X2 point (random variable).

    Returns
    -------
    C : int
        Relative class which point (X1,X2) belongs to.

    '''
    C = np.sign(-2*np.sign(X1)*abs(X1)**(2/3)+4*X2**2)
    return C

#np.random.seed(0)
#random.seed(7)
#generate uniformely distributed random variables X1 and X2
X1 = np.random.uniform(-1,1,5000)
X2 = np.random.uniform(-1,1,5000)

#Generate training set
N_train = np.zeros((1000,3),dtype=float)
for i in range(1000):
    N_train[i,0]=random.choice(X1)
    N_train[i,1]=random.choice(X2)
    N_train[i,2]=classe(N_train[i,0],N_train[i,1])

N_train_1=N_train[N_train[:,2]==1]
N_train_0=N_train[N_train[:,2]==-1]

#plot points with their respective classes: red C=-1, blu C=1
plt.figure()
plt.scatter(N_train_1[:,0],N_train_1[:,1],c='b',label='C=1')
plt.scatter(N_train_0[:,0],N_train_0[:,1],c='r',label='C=-1')
plt.legend(loc='upper left')
plt.xlim(left=-1,right=1)
plt.ylim(bottom=-1,top=1)
plt.title('Train set')
plt.show()

#generate test set
X1 = np.random.uniform(-1,1,5000)
X2 = np.random.uniform(-1,1,5000)
N_test = np.zeros((20000,3),dtype=float)
for i in range(20000):
    N_test[i,0]=random.choice(X1)
    N_test[i,1]=random.choice(X2)
    N_test[i,2]=classe(N_test[i,0],N_test[i,1])

N_test_1=N_test[N_test[:,2]==1] #points with class 1
N_test_0=N_test[N_test[:,2]==-1] #points with class -1

#plot points of test dataset with their respective estimated classes: red C=-1, blu C=1
plt.figure()
plt.scatter(N_test_1[:,0],N_test_1[:,1],c='b',label='C=1')
plt.scatter(N_test_0[:,0],N_test_0[:,1],c='r',label='C=-1')
plt.legend(loc='upper left')
plt.xlim(left=-1,right=1)
plt.ylim(bottom=-1,top=1)
plt.title('Test set')
plt.show()

#generate the decision tree classifier
clfX = tree.DecisionTreeClassifier(criterion='entropy')
#train the model
clfX = clfX.fit(N_train[:, 0:2],N_train[:,2])
#test the model
hatCtest = clfX.predict(N_test[:,0:2])
print('accuracy = ', accuracy_score(N_test[:,2],hatCtest))

#plot decision tree
feat_names=['X_1','X_2']
target_names=['C=-1','C=1']

#Function tree.export_graphviz() generates a GraphViz representation of the  
# decision tree, which is then written into out_file. Once exported, graphical
# renderings can be generated using, for example:
# $ dot -Tpng Tree.dot -o Tree.png    (PNG format)    

tree.export_graphviz(clfX,out_file='Tree.dot',
                      class_names='target_names',
                      filled=True, rounded=True,
                      special_characters=True)

