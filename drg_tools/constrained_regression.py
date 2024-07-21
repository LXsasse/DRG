
# Attempt to define a constrained regression with non-linear contraints on the parameters
# for example that parameters have to sum to 1
# Last time did not work for a toy problem: There are different optimization algorithms to choose from

import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from jax import grad, jacobian, hessian
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def correlation( b, xreal, yreal, regul):
    predy = np.dot(xreal, b)
    predyn = (predy.T - (1./len(predy)) * np.sum(predy, axis = 1)).T
    yrealn = (yreal.T - (1./len(predy)) * np.sum(yreal, axis = 1)).T
    predyn = (predyn.T/np.sqrt(np.sum(predyn*predy, axis = 1))).T
    yrealn = (yrealn.T/np.sqrt(np.sum(yrealn*yreal, axis = 1))).T
    return np.sum(predyn * yrealn)

def mse(b, xreal, yreal, regul):
    predy = np.sum(xreal*b)
    return 0.5*(1./len(xreal))*(np.sum((predy-yreal)**2) + regul *np.sum(b**2))

def abse(b, xreal, yreal, regul):
    predy = np.sum(xreal*b)
    return (1./len(xreal))*(np.sum(np.absolute(predy-yreal)) + regul *np.sum(b**2))

def eucnorm(b):
    norm2 = np.sum(b*b)
    return norm2
def absnorm(b):
    norm = np.sum(np.absolute(b))
    return norm

def sumnorm( b):
    norm = np.sum(b)
    return norm
def maxnorm(b):
    norm = np.amax(b)
    return norm

def null(b):
    return np.sum(0*b)

class constrained_regression():
    '''
    Attempt to define a constrained regression with a constrained parameter space.
    For example, the euclidean norm of the parameters should be smaller than 1
    Or the max of the parameters should be smaller than a given value. 
    '''
    def __init__(self, fit_intercept = True, objective = 'mse', constraint_up = 1., constraint_low = 0., constraint_func = '2-norm', ncores = 1, regular = 0., p_init = 'lsq'):
        
        self.init = p_init
        self.coef_ = None
        self.prediction_ = None
        self.error_ = None
        self.offset_ = None
        self.fit_intercept = fit_intercept
        self.lowerb = constraint_low
        self.upperb = constraint_up
        self.regular = regular
        self.cores = ncores
        
        if isinstance(objective, str):
            if objective == 'correlation':
                self.object_func = correlation
            elif objective == 'mse':
                self.object_func = mse
            elif objective == 'absolute':
                self.object_func = abse
            else: 
                print('Objective function not defined')
                sys.exit()
                
        else:
            self.object_func = objective
        
        if isinstance(constraint_func, str):
            if constraint_func == '2-norm':
                self.cons_f = eucnorm
            elif constraint_func == 'abs-norm':
                self.cons_f = absnorm
            elif constraint_func == 'sum':
                self.cons_f = sumnorm
            elif constraint_func == 'max':
                self.cons_f = maxnorm
            else:
                print('No constraint function defined')
                self.cons_f = null
        else:
            self.cons_f = constraint_func
    
    def predict(self,xr):
        if self.fit_intercept:
            xr = np.append(xr, np.ones(len(xr)).reshape(-1, 1), axis = 1)

        return np.dot(xr, self.coef_.T)
        
    def fit(self, X, Y):

        # define jacobian and hessian of objective function with autograd
        jacobian_  = grad(self.object_func, 0)
        def objectf_der(b, *kwargs):
            return jacobian_(b, *kwargs)

        hessian_ = hessian(self.object_func, 0)
        def objectf_hess(b, v, *kwargs):
            return np.dot(hessian_(b, *kwargs), v)
        
        # Using autograd to define jacobian and hessian
        jacoconst  = grad(self.cons_f)
        def cons_J(b):
            return jacoconst(b)
        
        # hessian
        hessconst = hessian(self.cons_f)
        def cons_H(b, v):
            #return v* hessconst(b)
            return v*np.zeros((np.shape(b)[0], np.shape(b)[0]))
        
        # nonlinear constraint defines a lower bound and upper bound of the funciton given(cons_f)
        #self.nonlinear_constraint = NonlinearConstraint(self.cons_f, self.lowerb, self.upperb, jac=cons_J, hess=cons_H)
        self.nonlinear_constraint = NonlinearConstraint(self.cons_f, self.lowerb, self.upperb)
        
        
        if self.fit_intercept:
            Xt = np.append(X, np.ones(len(X)).reshape(-1, 1), axis = 1)
        else:
            Xt = np.copy(X)
        
        if self.init == 'random':
            self.coef_ = np.random.random((np.shape(Xt)[1], np.shape(Y)[1]))
        elif self.init == 'lsq':
            self.coef_ = LinearRegression(fit_intercept = False).fit(Xt, Y).coef_
            print('Prelim coef_', np.sum(self.coef_), self.coef_)
        
        if len(np.shape(Y)) == 1:
            #output = minimize(self.object_func, self.coef_, args=(Xt, Y, self.regular), method='SLSQP', constraints =  self.nonlinear_constraint)
            output = minimize(self.object_func, self.coef_, args=(Xt, Y, self.regular), method='trust-constr', constraints =  self.nonlinear_constraint)
            #output = minimize(self.object_func, self.coef_, args=(Xt, Y, self.regular), method='trust-constr', jac=objectf_der, hessp=objectf_hess)
            #output = minimize(self.object_func, self.coef_, args=(Xt, Y, self.regular), method='trust-constr', jac=objectf_der, hessp=objectf_hess, constraints =  self.nonlinear_constraint)
            coef_ = output.x
            
        
        else:
            self.coef_ = np.zeros((np.shape(Xt)[1], np.shape(Y)[1]))
            coef_ = []
            if self.cores == 1:
                for i in range(np.shape(Y)[1]):
                    output = minimize(objectf, self.coef_[:,i], args=(Xt, Y[:,i], self.regular), method='trust-constr', jac=objectf_der, hessp=objectf_hess, constraints =  self.nonlinear_constraint)
                    coef_.append(output.x)
            else:
                coef_ = np.zeros(np.shape(self._coef_))
                def mini(i, X, Y):
                    output = minimize(objectf, self.coef_[:,i], args=(Xt, Y[:,i], self.regular), method='trust-constr', jac=objectf_der, hessp=objectf_hess, constraints =  self.nonlinear_constraint)
                    return i, output.x
                results = Parallel(n_jobs=num_cores)(delayed(mini)(i, Xt, Y) for i in range(np.shape(Y)[1]))
                for ri, res in results:
                    coef_[:,ri] = res
            
            coef_ = np.array(coef_)
            
        self.coef_ = coef_
        self.prediction_ = self.predict(X)
        
def scatter(x,y):
    fig=plt.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(x,y, c = 'c')
    lim = [min(np.amin(x), np.amin(y)), max(np.amax(x), np.amax(y))]
    ax.plot(lim, lim, c = 'maroon')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    return fig
    
if __name__ == '__main__':
    
    onp.random.seed(2)
    x = onp.random.randint(0,2,15).reshape(5,3)
    beta = (onp.random.random(3) - 0.5) * 2
    y = onp.sum(x*beta, axis = 1) 
    yn = y+ onp.random.random(len(x))-0.5
    scoef = onp.sum(beta)
    print('Original effects are', scoef, beta)
    
    lr = constrained_regression(fit_intercept = False, objective = 'mse', constraint_up = scoef, constraint_low = scoef, constraint_func = sum, ncores = 1, regular = 0., p_init = 'lsq')
    lr.fit(x,yn)
    coef = lr.coef_
    print('predicted coef are', onp.sum(coef), coef)
    ypred = lr.predict(x)
    
    lr = LinearRegression(fit_intercept = False)
    lr.fit(x,yn)
    ypredln = lr.predict(x)
    
    fig = scatter(y, yn)
    fig0 = scatter(y,ypred)
    fig1 = scatter(y,ypredln)
    
    plt.show()
    
    
    
    
    
    

