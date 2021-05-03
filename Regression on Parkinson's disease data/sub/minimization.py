# -*- coding: utf-8 -*-
"""
Sript minimization.py to be run with main.py
Use of class and subclasses

@author: Francesco Conforte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class SolveMinProb:
    """
    Class SolveMinProb includes two subclasses/algorityms: LLS (Linear Least Squares), 
    gradient algorithm. These algorithms are used to find
    the optimum solution w_hat to the minimization problem min_w ||y-Aw||^2.
    
    Inputs
    ------
    y : column vector with the regressand (Np rows)
        float
    A : Ndarray (Np rows Nf columns) with the Nf regressors for each of the Np points
        float

    Attributes
    ----------
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand, shape (Np,)
        float
    matr_val : Ndarray A, with the measured data (Np_val rows, Nf_val columns)
        float
    vect_val : y_val, Ndarray of known values of the regressand for validation, shape (Np_val,)
        float
    sol : w, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    Np_val : number of points in validation dataset (number of rows of A_val and y_val)
        integer
    min : value of ||y-Aw_hat||^2
        float
    err : Ndarray that stores the iteration step and corresponding value of ||y-Aw||^2 with the current w
        float  
    err_val : Ndarray that stores the iteration step and corresponding value of ||y_val-A_val@w||^2 with the current w
        float
        
    
    """
    def __init__(self,y=np.ones((3,1)),A=np.eye(3), y_val=np.ones((3,1)),A_val=np.eye(3)):
        self.matr=A # matrix with the measured data: Np rows/points and Nf columns/features
        self.Np=y.shape[0] # number of points 
        self.Nf=A.shape[1]# number of features/regressors
        self.Np_val=A_val.shape[0] #number of points in the validation dataset
        self.vect=y # regressand
        self.matr_val=A_val #validation matrix
        self.vect_val=y_val #regressand for validation dataset
        self.sol=np.zeros((self.Nf,),dtype=float) # unknown optimum weights w_hat (Nf elements)
        self.min=0.0 # obtained minimum value ||y-Aw_hat||^2
        self.err=np.zeros((100,2),dtype=float) # in case of iterative minimization, self.err 
        # stores the value of the function to be minimized, the first column stores the
        # iteration step, the second the corresponding value of the function
        self.err_val=np.zeros((100,2),dtype=float)
        return
    def plot_err(self,title='Square error',logy=0,logx=0):
        """ 
        plot_err plots the function to be minimized, at the various iteration steps

        Parameters
        ----------
        title : title of the plot
            string
        logy : 1 for logarithmic y scale, 0 for linear y scale
            integer
        logx : 1 for logarithmic x scale, 0 for linear x scale
            integer
        
        """
        err=self.err
        err_val=self.err_val
        it = np.count_nonzero(err, axis=0, keepdims=True)[0][1]
        plt.figure()
        if (logy==0) & (logx==0):
           plt.plot(err[0:it,0],err[0:it,1],label='training')
           plt.plot(err_val[0:it,0],err_val[0:it,1],label='validation')
        if (logy==1) & (logx==0):
            plt.semilogy(err[0:it,0],err[0:it,1],label='training')
            plt.semilogy(err_val[0:it,0],err_val[0:it,1],label='validation')
        if (logy==0) & (logx==1):
            plt.semilogx(err[0:it,0],err[0:it,1],label='training')
            plt.semilogx(err_val[0:it,0],err_val[0:it,1],label='validation')
        if (logy==1) & (logx==1):
            plt.loglog(err[0:it,0],err[0:it,1],label='training')
            plt.loglog(err_val[0:it,0],err_val[0:it,1],label='validation')
        plt.xlabel('n')
        plt.ylabel('MSE(n)')
        plt.title(title)
        plt.margins(0.01,0.1)# leave some space between the max/min value and the frame of the plot
        plt.grid()
        plt.legend()
        plt.show()
        return
    def print_result(self,title,e,y_norm,sy,my):
        """ 
        print_result prints a dataframe containing parameters: mean, standard 
        deviation, mean squared value, parameter R2 of the error in each of the
        three subsets

        Parameters
        ----------
        title: tipically the algorithm used to find w_hat
            string
            
        e: Nd array containing the errors for training, validation and test datasets
            float
            
        y_norm: Nd array containing normalized regressand values for training, validation and test dataset
            float
            
        sy: standard deviation
            float
        
        my: mean 
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
    
    def plot_w_hat(self,X_tr_norm,title):
        """ 
        plot_w_hat plots w_hat (optimum solution of the minimization problem) 

        Parameters
        ----------
            
        X_tr_norm : DataFrame containing the features 
            DataFrame
            
        title : typically the algorithm used to find w_hat, it is the title of the figure
            string
        """
        w_hat=self.sol
        regressors=list(X_tr_norm.columns)
        Nf=len(w_hat)
        nn=np.arange(Nf)
        plt.figure(tight_layout=True)
        plt.plot(nn,w_hat,'-o')
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)#, **kwargs)
        plt.ylabel('w_hat(n)')
        plt.title(title)
        plt.margins(0.01,0.1)# leave some space 
        plt.grid()
        plt.show()
        
    def plot_hist(self, title, e):
        """
        plot_hist plots the estimation error for training, validation 
        and test datasets useful to check if the regression error shows 
        peculiar trends, which might reveal an error in the script

        Parameters
        ----------
        title : tipically the algorithm used to find w_hat, it's the title of the figure
            string
         
        e: Nd array containing the errors for training, validation and test datasets
            float
        Returns
        -------
        None.

        """
        
        plt.figure()
        plt.hist(e,bins=50,density=True, histtype='bar',
        label=['training','validation','test'])
        plt.xlabel('y-y_hat')
        plt.ylabel('P(error in bin)')
        plt.legend()
        plt.grid()
        plt.title('Error histograms: ' + title)
        plt.show()
        
    def plot_y_hat_vs_y(self, title='test',y_te=np.ones((3,1)),y_hat_te=np.ones((3,1))):
        """
        plot_y_hat_vs_y plots the comparison the true and the regressed value of

        Parameters
        ----------
        title : tipically the algorithm used to find w_hat, it's the title of the figure
            string
            
        y_te : x axis: column vector with the regressand of the test (Np_test rows)
            Nd array of float
            
        y_hat_te : y axis column vector with the estimated regressand (Np_test rows)
            Nd array of float

        Returns
        -------
        None.

        """
        plt.figure()
        plt.plot(y_te,y_hat_te,'.')
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.xlabel('y')
        plt.ylabel('yhat')
        plt.grid()
        plt.title(title)
        plt.show()
        
class SolveLLS(SolveMinProb):
    """ 
    Linear least squares: the optimum solution of the problem ||y-Aw||^2 is the
    w_hat=(A^TA)^{-1}A^Ty
    
    Inputs (inherited from class SolveMinProb)
    ------
    y : column vector with the regressand (Np rows)
        float
    A : Ndarray (Np rows Nf columns) 
    
    
    Methods
    -------
    run : runs the method and stores the optimum weights w in self.sol
            and the corresponding objective function in self.min
            
    Methods (inherited from class SolveMinProb)
    -------
    print_result : prints the optimum weights w_hat and the corresponding objective function
    plot_w_hat: plots the solution w_hat
            
    
    Attributes (inherited from class SolveMinProb)
    ----------
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand (Np elements)
        float
    sol : w_hat, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    min : value of ||y-Aw_hat||^2
        float
    err : not used, set to scalar 0.0
        float   
        
        
    Example: 
    -------
    lls=SolveLLS(y,A)    
    lls.run()    
    lls.print_result('Linear Least Squares')
    lls.plot_w_hat('Linear Least Squares')
        
    """
    
    def run(self):
        A=self.matr
        y=self.vect
        w_hat=np.linalg.inv(A.T@A)@(A.T@y)
        self.sol=w_hat
        self.min=np.linalg.norm(A@w_hat-y)**2
        self.err=0
        return self.sol
        
    
class SolveStochWithADAM(SolveMinProb):
    
    """ Stochastic gradient algorithm with ADAM optimization: 
        the optimum solution of the problem ||y-Aw||^2 is obtained through
        the stochastic gradient algorithm by considering each time a row of 
        the matrix A and by considering not the gradient of each step but 
        an estimation of the mean of all the previous gradient of the previous
        steps (ADAM optimization)
    
    Methods
    -------
    run : runs the method and stores the optimum weights w in self.sol
            and the corresponding objective function in self.min
            Input parameters:  learning coeff gamma (default 1e-3),
            number of iterations Nit (default 100),
            coefficient beta for k=1 (default 0.99),
            coefficient beta for k=2 (default 0.999),
            coefficient epsilon for avoiding division by 0 (default 1e-8)
            

    Attributes (inherited from class SolveMinProb)
    ----------
        
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand, shape (Np,)
        float
    matr_val : Ndarray A, with the measured data (Np_val rows, Nf_val columns)
        float
    vect_val : y_val, Ndarray of known values of the regressand for validation, shape (Np_val,)
        float
    sol : w, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    Np_val : number of points in validation dataset (number of rows of A_val and y_val)
        integer
    min : value of ||y-Aw_hat||^2
        float
    err : Ndarray that stores the iteration step and corresponding value of ||y-Aw||^2 with the current w
        float  
    err_val : Ndarray that stores the iteration step and corresponding value of ||y_val-A_val@w||^2 with the current w
        float
    
    """
    
    def run(self, gamma=1e-3, Nit=100, b1 = 0.99, b2 = 0.999, epsilon = 1e-8,sy=0):
        self.err=np.zeros((Nit,2),dtype=float)
        self.err_val=np.zeros((Nit,2),dtype=float)
        self.gamma=gamma
        self.Nit=Nit
        A=self.matr
        y=self.vect
        A_val=self.matr_val
        y_val=self.vect_val
        w=np.random.rand(self.Nf,)# random initialization of the weight vector
        n = 0
        mu1_hat = 0
        mu2_hat = 0
        
        for it in range(Nit):
            grad = 2*(A[n, :].T@w - y[n])*A[n, :]
            mu1_hat = b1 * mu1_hat + (1-b1)*grad
            mu2_hat = b2 * mu2_hat + (1-b2)*grad**2
            mu1_hat_hat = mu1_hat/(1-b1**(it+1))
            mu2_hat_hat = mu2_hat/(1-b2**(it+1))
            mod_grad = np.divide(mu1_hat_hat, np.sqrt(mu2_hat_hat + epsilon))
            w=w-gamma*mod_grad
            self.err[it,0]=it
            self.err[it,1]=np.mean((A@w-y)**2)
            self.err_val[it,0]=it
            self.err_val[it,1]=np.mean((A_val@w-y_val)**2)
            n+=1
            if n >= np.shape(A)[0]:
                n = 0
            
        self.sol=w
        self.min=self.err[it,1]
        return self.sol

class SolveRidge(SolveMinProb):
    """
    Ridge Regression: the optimum solution of the problem        
    ||y-Aw||^2 + lambda*||w|| 
    is the w_hat=(A^TA+lambda*I)^{-1}A^Ty
    
    Inputs (inherited from class SolveMinProb)
    ------
    y : column vector with the regressand (Np rows)
        float
    A : Ndarray (Np rows Nf columns) 
    
    
    Methods
    -------
    run : runs the method and stores the optimum weights w in self.sol
    plot_MSEvsLambda : plots the MSE vs lambda for training and validation
            
    Methods (inherited from class SolveMinProb)
    -------
    print_result : prints the optimum weights w_hat and the corresponding objective function
    plot_w_hat: plots the solution w_hat
            
    
    Attributes (inherited from class SolveMinProb)
    ----------
    matr : Ndarray A, with the measured data (Np rows, Nf columns)
        float
    vect : y, Ndarray of known values of the regressand, shape (Np,)
        float
    matr_val : Ndarray A, with the measured data (Np_val rows, Nf_val columns)
        float
    vect_val : y_val, Ndarray of known values of the regressand for validation, shape (Np_val,)
        float
    sol : w, Ndarray with shape (Nf,), optimum value of vector w that minimizes ||y-Aw||^2
        float
    Np : number of available points (number of rows of A and y)
        integer
    Nf : number of available features (number of columns of A)
        integer
    Np_val : number of points in validation dataset (number of rows of A_val and y_val)
        integer
    err : Ndarray that stores the iteration step and corresponding value of ||y-Aw||^2 with the current w
        float  
    err_val : Ndarray that stores the iteration step and corresponding value of ||y_val-A_val@w||^2 with the current w
        float
       
    """
    def run(self, _lambda):
        A=self.matr
        y=self.vect
        A_val=self.matr_val
        y_val=self.vect_val  
        I = np.identity(self.Nf)
        w_hat=np.linalg.inv(A.T@A + _lambda*I)@(A.T@y)
        self.sol=w_hat
        return self.sol
    
    
    def plot_MSEvsLambda(self, errors_tr, errors_val, best, title='Mean Square Error vs lambda'):
        plt.figure()
        plt.title("RIDGE: MSE with different lambdas")
        plt.ylabel("MSE")
        plt.xlabel("Lambda")
        #extraticks = [0,10,20,29,30,40,50,60,70,80,90,100]
        plt.xticks(np.arange(0, len(errors_val), step=10))
        #plt.xticks(extraticks)
        plt.plot(errors_tr[:,0],errors_tr[:,1], label='training')
        plt.plot(errors_val[:,0],errors_val[:,1], label='validation')
        plt.plot(best, errors_val[best,1], marker='x', label='best lambda')
        plt.grid()
        plt.legend()
        plt.show()
        return
    
# class SolveGrad(SolveMinProb):
#     """ Gradient algorithm: the optimum solution of the problem ||y-Aw||^2 is obtained through
#     the gradient algorithm

#     Methods
#     -------
#     run : runs the method and stores the optimum weights w in self.sol
#             and the corresponding objective function in self.min
#             Input parameters:  learning coeff gamma (default 1e-3), number of iterations Nit (def 100)
            

#     Attributes (inherited from class SolveMinProb)
#     ----------
#     matr : Ndarray A, with the measured data (Np rows, Nf columns)
#         float
#     vect : y, Ndarray of known values of the regressand (Np elements)
#         float
#     gamma : learning coefficient
#         float
#     Nit : number of iterations
#         integer
#     sol : w_hat, Ndarray (Nf elements), optimum value of vector w that minimizes ||y-Aw||^2
#         float
#     Np : number of available points (number of rows of A and y)
#         integer
#     Nf : number of available features (number of columns of A)
#         integer
#     min : value of ||y-Aw_hat||^2
#         float
#     err : not used, set to scalar 0.0
#         float   

#     Example: 
#     -------
#     ga=SolveGrad(y,A)
#     gamma=1e-3
#     Nit=100    
#     ga.run(gamma,Nit)    
#     ga.print_result('Gradient algorithm')
#     ga.plot_w_hat('Gradient algorithm')


#     """
    
#     def run(self,gamma=1e-3,Nit=100):
#         self.err=np.zeros((Nit,2),dtype=float)
#         self.gamma=gamma
#         self.Nit=Nit
#         A=self.matr
#         y=self.vect
#         w=np.random.rand(self.Nf,)# random initialization of the weight vector
#         for it in range(Nit):
#             grad=2*A.T@(A@w-y)
#             w=w-gamma*grad
#             self.err[it,0]=it
#             self.err[it,1]=np.mean((A@w-y)**2)
#         self.sol=w
#         self.min=self.err[it,1]

# class SolveSteepest(SolveMinProb):
    
#     """ Steepest Descent algorithm: the optimum solution of the problem ||y-Aw||^2 is obtained through
#     the gradient algorithm (optimum step)
    
#     Methods
#     -------
#     run : runs the method and stores the optimum weights w in self.sol
#             and the corresponding objective function in self.min
#             Input parameters:  learning coeff gamma (default 1e-3), number of iterations Nit (def 100)
            

#     Attributes (inherited from class SolveMinProb)
#     ----------
#     matr : Ndarray A, with the measured data (Np rows, Nf columns)
#         float
#     vect : y, Ndarray of known values of the regressand (Np elements)
#         float
#     gamma : learning coefficient
#         float
#     Nit : number of iterations
#         integer
#     sol : w_hat, Ndarray (Nf elements), optimum value of vector w that minimizes ||y-Aw||^2
#         float
#     Np : number of available points (number of rows of A and y)
#         integer
#     Nf : number of available features (number of columns of A)
#         integer
#     min : value of ||y-Aw_hat||^2
#         float
#     err : not used, set to scalar 0.0
#         float
#     """
    
#     def run(self, gamma=1e-3, Nit = 100):
#         self.err = np.zeros((Nit, 2), dtype=float)
#         A = self.matr
#         y = self.vect
#         #Hessian Matrix. It does not depend on w because the Hessian matrix of the 
#         #square error is 2*A.T@A, so it's out of the loop
#         H = 2 * A.T@A
#         w = np.random.rand(self.Nf,) #random initialization of w
        
#         for it in range(Nit):
#             grad=2*A.T@(A@w-y)
#             if not grad.any(): #to check wheter grad is 0. 
#                 break
#             gamma = np.linalg.norm(grad)**2/(grad.T@H@grad)
#             w = w - gamma * grad
#             self.err[it, 0] = it
#             self.err[it, 1] = np.linalg.norm(A@w-y)**2
        
#         #assign the optimum vector w
#         self.sol=w
#         self.min=self.err[it-2, 1]
#         return
            
            
