import sys, os 
import numpy as np
import scipy.stats as stats
from scipy.sparse.linalg import svds
from sklearn import linear_model, metrics
from sklearn.decomposition import SparsePCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from functools import reduce
from functions import mse, correlation
from data_processing import create_outname, check, numbertype, manipulate_input, readin, create_sets
from output import save_performance, print_averages, plot_scatter
from torch_regression import torch_Regression


# Logistic regression wrapper that uses joblib to fit outclasses independently
class logistic_regression():
    def __init__(self, n_jobs = None, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, l1_ratio=None):
        
        if n_jobs is None:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        
        self.model = linear_model.LogisticRegression(n_jobs = 1, penalty=penalty, dual=dual, tol=tol, C=1./C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, l1_ratio=l1_ratio)
        
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def get_params(self):
        return self.__dict__

    def set_params(self, **kwargs):
        for kwar in kwargs:
            if str(kwar) in self.__dict__.keys():
                setattr(self, kwar, kwargs[kwar])
            if str(kwar) in self.model.get_params().keys():
                self.model.set_params(**{str(kwar):kwargs[kwar]})
            if str(kwar) == 'alpha':
                self.model.set_params(C = 1./kwargs[kwar])
        
    def fit_single(self, X, Y, i, sample_weight = None):
        self.model.fit(X,Y, sample_weight = sample_weight)
        coef = self.model.coef_[0]
        if self.fit_intercept:
            intercept = self.model.intercept_[0]
        else:
            intercept = None
        return coef, intercept, i
    
    def fit(self, X, Y, sample_weight = None):
        self.coef_ = None
        self.intercept_ = None
        
        if len(np.shape(Y)) ==1:
            self.model.set_params(n_jobs = self.n_jobs)
            self.model.fit(X,Y, sample_weight = sample_weight)
            self.coef_ = self.model.coef_[0]
            if self.fit_intercept:
                self.intercept_ = self.model.intercept_
        else:
            self.coef_ = np.zeros((np.shape(Y)[-1], np.shape(X)[-1]))
            if self.fit_intercept:
                self.intercept_ = np.zeros(np.shape(Y)[-1])
            results = Parallel(n_jobs=self.n_jobs)(delayed(self.fit_single)(X, Y[:,i], i, sample_weight=sample_weight) for i in range(np.shape(Y)[-1]))
            for rc, ri, r in results:
                self.coef_[r] = rc
                if ri is not None:
                    self.intercept_[r] = ri
                
    def predict(self,X):
        pred = (self.predict_proba(X) > 0).astype(int)
        return pred
    
    def predict_proba(self,X):
        pred = np.dot(X,self.coef_.T)
        if self.fit_intercept:
            pred += self.intercept_
        return pred
    

    
    
class sk_regression():
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=None, tol=None, solver='auto', positive=False, random_state=None, penalty = None, pca = None, center = False, optimize_alpha = True, change_alpha =.6, validation_axis = 1, alpha_search = 'dependent', normalize = False, full_nonlinear = False, logistic = False, warm_start = False, verbose = True, n_jobs = None, refit_l1 = False, **kwargs):
        
        self.refit_l1 = refit_l1 # l1 regression selects features but then regression is fit to selected subset to rescale coefficients
        
        self.verbose = verbose
        self.penalty = penalty # regularization loss
        self.alpha=alpha # regularization weight
        
        self.optimize_alpha = optimize_alpha # define whether one alpha should be iteratively increased or decreased until performance gets worse
        self.change_alpha = change_alpha # value by which alpha is reduces or increased in each iteration
        self.alpha_search = alpha_search # how to look for alpha, independent fits model for each class seperately to avoid issues with convergence when fitting across several classes, 'dependent' is faster but might run into issues for cases where one classe prevents convergence.
        self.validation_axis = validation_axis # if validation set is provided, axis along which the auc or correlation is computed
        
        self.solver=solver # method to find optimal solution
        self.max_iter=max_iter # for iteratively updating algorithms
        self.tol=tol # checks the dual gap for optimality and continues until it is smaller than tol
        self.positive=positive # restricts parameters to positive values
        self.random_state=random_state # seed for np.random
        self.warm_start = warm_start # after restart, can be used keep parameters learned in last round
        
        if n_jobs is None:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        
        self.pca = pca # if not None the input dimensions are reduced to explained variance or number of principal components
        
        if isinstance(pca, int) or isinstance(pca, float): 
            self.center = True
            self.fit_intercept = False
        else:
            self.center = center
            self.fit_intercept=fit_intercept
        
        self.normalize = normalize # for some tasks it might be important to normalize features to the same variance
        self.full_nonlinear = full_nonlinear # creates non-linear features from multiplying input feautures
        
        self.logistic = logistic
        
        self.Xm = None # Mean of X
        self.Xs = None # Std of X
        self.Xmask = None
        self.coef_ = None # Coefficients of the model
        self.V = None # Singular vectors of X
        self.Vinv = None # Pseudo inverse of singular vectors
        
        self.z_scores = None
        self.p_values = None
        self.select_mask = None
        
        if self.solver in ['SGD', 'Adam']:
            self.logit = torch_Regression()
        else:
            if logistic:
                # Use logistic regression
                if solver == 'auto' or solver is None:
                    self.solver = 'saga'
                if max_iter is None:
                    self.max_iter = 300
                if tol is None:
                    self.tol = 0.001
                
                self.logit = logistic_regression(penalty = self.penalty, C=self.alpha, solver = self.solver, fit_intercept=self.fit_intercept, warm_start = self.warm_start, n_jobs = self.n_jobs, max_iter = self.max_iter, tol = self.tol)
            else:
                if penalty is None:
                    self.penalty = 'none'
                    self.logit = linear_model.LinearRegression(fit_intercept=self.fit_intercept, n_jobs = self.n_jobs, positive = self.positive)
                
                elif penalty == 'l2':
                    if tol is None:
                        self.tol = 0.001
                        
                    self.logit = linear_model.Ridge(alpha=alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, solver=self.solver, positive=self.positive, random_state = self.random_state)
                
                elif penalty == 'l1':
                    if max_iter is None:
                        self.max_iter = 2000
                    if tol is None:
                        self.tol = 0.0001
                    self.logit = linear_model.Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, positive=self.positive, random_state=self.random_state, warm_start = self.warm_start)
                
                elif penalty == 'elastic':
                    if max_iter is None:
                        self.max_iter = 1000
                    if tol is None:
                        self.tol = 0.0001
                    self.logit = linear_model.ElasticNet(alpha=alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, positive=self.positive, random_state = self.random_state, selection = 'random')
                
                else:
                    self.penalty = 'none'
                    self.logit = linear_model.LinearRegression(fit_intercept=self.fit_intercept, n_jobs = self.n_jobs, positive = self.positive)
        
    def transform(self, X):
        # Could explore and SVD algo where the ratio of the next singular value to the rest determines if another one will be added.
        
        if self.Xmask is not None:
            X = X[:,self.Xmask]
        
        if (self.center or self.normalize) and self.Xm is None:
            self.Xm = np.mean(X, axis = 0)
        
        if self.center or self.normalize:
            X = X - self.Xm
        
        if self.normalize and self.Xs is None:
            self.Xs = np.std(X, axis = 0)
            self.Xmask = self.Xs != 0
            if (~self.Xmask).any():
                self.Xs = self.Xs[self.Xmask]
                self.Xm = self.Xm[self.Xmask]
                X = X[:,self.Xmask]
            else:
                self.Xmask = None
        
        if self.normalize :
            X = X/self.Xs
            X = np.nan_to_num(X)

        
        
        if self.pca is not None and self.Vinv is None:
            if self.pca < 1:
                self.pca = int(np.amin(np.shape(X))*self.pca)
            else:
                self.pca = int(self.pca)
            u,s,v = svds(X, k=self.pca, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, solver='arpack')
            sorts = np.argsort(-s)
            u,s,v = u[:, sorts],s[sorts],v[sorts]
            
            self.Vinv = np.linalg.pinv(v)
            self.V = v
        if self.pca is not None:
            X = np.dot(X,self.Vinv)
            
        return X
    
    
    
    def optimize_hyperparameter(self, X, Y, Xval, Yval, inc, axis = 1, weights = None, method = 'independent', min_alpha = 1e-11, max_alpha = 1e3):
        max_alpha, min_alpha = self.alpha*max_alpha, self.alpha*min_alpha
        
        def fit_single(X, Y, i, sample_weight = None):
            model = self.logit
            model.fit(X,Y, sample_weight = sample_weight)
            coef = model.coef_
            if self.fit_intercept:
                intercept = model.intercept_
                #print(intercept,i, metrics.roc_auc_score(Y,model.predict_proba(X).flatten()))
            else:
                intercept = None
            return coef, intercept, i
        
        if axis == 0 and np.shape(Yval)[1]>2:
            self.logit.fit(X, Y, sample_weight = weights)
            
            self.coef_ = self.logit.coef_.T
            if self.fit_intercept:
                self.coef_ = np.append(self.coef_, [self.logit.intercept_], axis = 0)
            
            if self.logistic:
                pred = self.logit.predict_proba(Xval)
                perform = 1.-np.mean(metrics.roc_auc_score(Yval.T, pred.T, average = None))
            else:
                pred = self.logit.predict(Xval)
                perform = 1.-np.mean(self.correlation(pred, Yval,axis = 1))
            
            alpha = np.copy(self.alpha)*inc
            stdval = np.std(Yval, axis = 1)
            j = 0
            while True:
                self.logit.set_params(alpha = alpha)
                self.logit.fit(X, Y,sample_weight = weights)
                
                if self.logistic:
                    pred = self.logit.predict_proba(Xval)
                    nperform = 1.-np.mean(metrics.roc_auc_score(Yval.T, pred.T, average = None))
                else:
                    pred = self.logit.predict(Xval)
                    nperform = 1.-np.mean(self.correlation(pred, Yval,axis = 1))
                
                stdpred = np.std(pred, axis = 1)
                if (stdpred/stdval).any() < 1e-8:
                    alpha = alpha*inc
                    j += 1
                elif nperform < perform:
                    alpha *= inc
                    j += 1
                    self.coef_ = self.logit.coef_.T
                    if self.fit_intercept:
                        self.coef_ = np.append(self.coef_, [self.logit.intercept_], axis = 0)
                elif nperform >= perform and j == 0:
                    inc = 1./inc
                elif nperform > perform:
                    break 
                    
        else:
            self.balpha_list = np.ones(len(Y[0]))*self.alpha
            j = 0
            updown = -np.ones(len(Y[0]))
            hasnotimproved = np.zeros(len(Y[0]))
            hasincreased = np.zeros(len(Y[0]))
            js = np.zeros(len(Y[0]),dtype = int)
            if method == 'independent':
                if 'n_jobs' in self.logit.get_params().keys():
                    self.logit.set_params(n_jobs = 1)
                results = []
                self.coef_ = np.zeros((np.shape(Y)[-1], np.shape(X)[-1]+int(self.fit_intercept)))
                results = Parallel(n_jobs=self.n_jobs)(delayed(fit_single)(X, Y[:,i], i, sample_weight = weights) for i in range(np.shape(Y)[-1]))
                for rc, ri, r in results:
                    self.coef_[r, :np.shape(X)[1]] = rc
                    if self.fit_intercept:
                        self.coef_[r, -1] = ri
            else:
                self.logit.fit(X, Y, sample_weight = weights)
                self.coef_ = self.logit.coef_
                if len(np.shape(self.coef_)) == 1:
                    self.coef_ = self.coef_.reshape(1,-1)
                
                coefnum = np.sum(self.coef_!=0, axis = 1)
                if self.fit_intercept:
                    self.coef_ = np.append(self.coef_, self.logit.intercept_.reshape(-1,1),axis = 1)
            
            pred = np.dot(Xval,self.coef_[:,:np.shape(Xval)[1]].T)
            if self.fit_intercept:
                pred += self.coef_[:,-1]
            if self.logistic:
                perform = 1.-metrics.roc_auc_score(Yval, pred, average = None)
            else:
                perform = 1.-self.correlation(pred, Yval, axis = 0)
            alpha = np.copy(self.alpha) * inc
            goesup = True
            currstage = -1
            nperform = np.copy(perform)
            while True:
                j += 1
                self.logit.set_params(alpha = alpha)
                updownmask = updown != 0
                
                if self.verbose:
                    print(alpha, int(np.sum(updown==0)), int(np.sum(updown==currstage)), round(np.mean(perform),3))
                    #print(hasincreased, hasnotimproved)
                if method == 'independent':
                    nupdown = int(np.sum(updownmask))
                    iupdown = np.where(updownmask)[0]
                    coef_ = np.zeros((nupdown, np.shape(X)[-1]+int(self.fit_intercept)))
                    results = Parallel(n_jobs=self.n_jobs)(delayed(fit_single)(X, Y[:,iupdown[i]], i, sample_weight = weights) for i in range(nupdown))
                    for rc, ri, r in results:
                        coef_[r, :np.shape(X)[1]] = rc
                        if self.fit_intercept:
                            coef_[r, -1] = ri
                else:
                    self.logit.fit(X, Y[:,updownmask], sample_weight = weights)
                    coef_ = self.logit.coef_
                    if len(np.shape(coef_)) == 1:
                        coef_ = coef_.reshape(1,-1)
                    if self.fit_intercept:
                        coef_ = np.append(coef_, self.logit.intercept_.reshape(-1,1),axis = 1)
                opred = np.dot(Xval,coef_[:,:np.shape(Xval)[1]].T)
                if self.fit_intercept:
                    opred += coef_[:,-1]
                pred[:,updownmask] = opred
                coefnum[updownmask] = np.sum(coef_[:,:np.shape(Xval)[1]] != 0,axis = 1)
                if self.logistic:
                    nperform = 1.-metrics.roc_auc_score(Yval, pred, average = None)
                else:
                    nperform = 1.-self.correlation(pred, Yval, axis = 0)
                #print(nperform[updownmask], perform[updownmask], js[updownmask])
                for nf, nperf in enumerate(nperform):
                    if updownmask[nf]:
                        if coefnum[nf] == 0 and goesup:
                            updown[nf] = -1
                            js[nf] = 0
                        elif goesup == False and coefnum[nf] == 0:
                            updown[nf] = 0
                        elif nperf < perform[nf]:
                            updown[nf] = currstage
                            js[nf] += 1
                            self.coef_ [nf] = coef_[nf-int(np.sum(~updownmask[:nf]))]
                            self.balpha_list[nf] = alpha
                            perform[nf] = nperf
                            hasnotimproved[nf] = 0
                            hasincreased[nf] = 0
                        elif nperf > perform[nf] and j == 1:
                            updown[nf] = 1
                            js[nf] += 1
                            self.coef_ [nf] = coef_[nf-int(np.sum(~updownmask[:nf]))]
                            self.balpha_list[nf] = alpha
                        elif coefnum[nf] > 0 and js[nf] == 0 and coefnum[nf] <= np.sum(self.coef_[nf,:np.shape(Xval)[1]]!=0):
                            updown[nf] = currstage
                            self.coef_ [nf] = coef_[nf-int(np.sum(~updownmask[:nf]))]
                            self.balpha_list[nf] = alpha
                        elif nperf == perform[nf] and js[nf] >= 1 and updown[nf] == currstage:
                            hasnotimproved[nf] +=1
                        elif nperf > perform[nf] and js[nf] >= 1 and updown[nf] == currstage:
                            hasincreased[nf] +=1
                        if hasnotimproved[nf] == 10 or hasincreased[nf] == 3:
                            updown[nf] = 0
                            #print( nf, alpha, perform[nf])
                        
                
                if alpha <= min_alpha:
                    if (updown == 1).any() and goesup:
                        self.coef_ [updownmask] = coef_
                        self.balpha_list[updownmask] = alpha
                        currstage = 1
                        updown[updown == -1] = 0
                        alpha = np.copy(self.alpha)/inc
                        goesup = False
                    else:
                        self.coef_ [updownmask] = coef_
                        self.balpha_list[updownmask] = alpha
                        self.coef_ = self.coef_.T
                        print('Aborted at ', alpha)
                        break
                elif alpha >= max_alpha:
                    self.coef_ [updownmask] = coef_
                    self.balpha_list[updownmask] = alpha
                    self.coef_ = self.coef_.T
                    print('Aborted at ', alpha)
                    break
                elif (updown == -1).any():
                    alpha = alpha*inc
                elif (updown == 1).any() and goesup:
                    currstage = 1
                    alpha = np.copy(self.alpha)/inc
                    goesup = False
                elif (updown == 1).any():
                    currstage = 1
                    alpha = alpha/inc
                elif (updown == 0).all():
                    self.coef_ = self.coef_.T
                    for p, per in enumerate(perform):
                        print('OUTclass', p, per, self.balpha_list[p], len(np.nonzero(self.coef_[:,p])[0]))
                    break
                
    def fit(self, X, Y, XYval = None, weights = None):
        # X given as N-datapoints X N_features
        self.shapeX = np.shape(X)
        self.shapeY = np.shape(Y)
        
        if self.full_nonlinear:
            self.non_linearlist = []
            for i in range(len(X[0])):
                for j in range(i+1, len(X[0])):
                    self.non_linearlist.append([i,j])
            X = self.combine_nonlinear(X)
            print(np.shape(X))
        
        X = self.transform(X)
        
        if self.optimize_alpha and XYval is not None and self.penalty != 'none':
            # fit each class independently with reducing or increasing alpha
            # check for best hyperparameter with validation set
            Xval = XYval[0]
            Yval = XYval[1]
            if self.full_nonlinear:
                Xval = self.combine_nonlinear(Xval)
            Xval = self.transform(Xval)

            self.optimize_hyperparameter(X, Y, Xval, Yval, self.change_alpha, axis = self.validation_axis, weights = weights, method = self.alpha_search)
            
            if self.pca is not None:
                if self.fit_intercept:
                    self.coef_ = np.append(np.dot(self.Vinv, self.coef_[:-1]), self.coef_[[-1]],axis = 1)
                else:
                    self.coef_ = np.dot(self.Vinv, self.coef_)
                self.pca = None
        else:
            self.logit = self.logit.fit(X, Y, sample_weight= weights)
        
            if len(np.shape(self.coef_)) == 1:
                self.coef_ = self.coef_.reshape(1, -1)
            self.coef_ = self.logit.coef_.T
            
            if self.pca is not None:
                # X = U*S*V <==> U*S = X*pinv(V) and Y^ = U^*S^*beta = X*pinv(V)*beta ==> beta^ = pinv(V)*beta
                self.coef_ = np.dot(self.Vinv, self.coef_)
                self.pca = None # SET back to None because input does not need to be transformed anymore
            
            if self.fit_intercept:
                self.coef_ = np.append(self.coef_,[self.logit.intercept_], axis = 0)
        
        self.coef_ = self.coef_.T
        
        if self.penalty == 'l1' and self.refit_l1:
            self.logit = linear_model.LinearRegression(fit_intercept=self.fit_intercept, positive = self.positive)
            for c, coef in enumerate(self.coef_):
                self.logit.fit(X[:,coef[:len(X[0])]!=0],Y[:,c])
                self.coef_[c][:len(X[0])][coef[:len(X[0])]!=0] = self.logit.coef_
                if self.fit_intercept:
                    self.coef_[c][-1] = self.logit.intercept_
            
        
    def combine_nonlinear(self, X):
        lenset = len(self.non_linearlist)
        nonlX = np.zeros((len(X), lenset))
        for c, comb in enumerate(self.non_linearlist):
            nonlX[:, c] = X[:,comb[0]]*X[:,comb[1]]
        return np.append(X[:, np.unique(np.concatenate(self.non_linearlist))], nonlX, axis = 1)
        
        
    def predict(self, X):
        
        if self.full_nonlinear:
            X = self.combine_nonlinear(X)
        
        X = self.transform(X)
        
        if self.fit_intercept:
            X = np.append(X, np.ones((len(X),1)), axis = 1)
            
        return np.dot(X,self.coef_.T)
    
    def mse(self, y1, y2, axis = None):
        return np.sum((y1-y2)**2, axis = axis)
    
    def correlation(self, y1, y2, axis = 1):
        mean1, mean2 = np.mean(y1, axis = axis), np.mean(y2, axis = axis)
        if axis ==1:
            y1mean, y2mean = y1-mean1[:,None], y2-mean2[:,None]
        else:
            y1mean, y2mean = y1-mean1, y2-mean2
        n1, n2 = np.sqrt(np.sum(y1mean**2, axis = axis)), np.sqrt(np.sum(y2mean**2, axis = axis))
        n12 = n1*n2
        y12 = np.sum(y1mean*y2mean, axis = axis)
        y12[n12 == 0] = -1
        n12[n12 == 0] = 1
        corout = y12/n12
        corout[np.isnan(corout)] = -1
        corout[np.isinf(corout)] = -1
        return np.around(corout,4)        
    
    def assess_impact(self, featurelist, X, Y, accuracy_func = 'mse'):
        # repeat fit without feature and measure difference to performance from entire model
        
        if featurelist == 'ALL':
            featurelist = np.arange(self.shapeX[-1], dtype = int)
        yref = self.predict(X)
        
        if accuracy_func == 'mse':
            acfunc = self.mse
        elif accuracy_func == 'correlation':
            acfunc = self.mean_correlation
            
        ref_accuracy = acfunc(yref, Y)
        impact_list = []
        for f in featurelist:
            # retrain model and make predicton
            reduced_model = sk_regression(alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=self.copyX, max_iter=self.max_iter, tol=self.tol, solver=self.solver, positive=self.positive, random_state=self.random_state, penalty = self.penalty, pca = self.pca, center = self.center)
            reduced_X_train = np.delete(self.X, f, axis = 1)
            reduced_model.fit(reduced_X_train, Y)
            reduced_X_test = np.delete(X, f, axis = 1)
            reduced_ypred = reduced_model.predict(reduced_X_test)
            reduced_accuracy = acfunc(reduced_ypred, Y)
            impact_list.append(reduced_accuracy-ref_accuracy)
        return impact_list
    
    def reduced_coef(self, max_feat = 2000):
        if len(self.coef_[0]-int(self.fit_intercept)) > max_feat:
            self.select_mask = np.where(np.sum(np.absolute(self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)]),axis = 0) > 0)[0]
            if len(self.select_mask) > max_feat:
                # choose features with highest mean values
                meancoef = -np.sum(np.absolute(self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)]),axis = 0)
                self.select_mask = np.argsort(meancoef)[:max_feat]
                #print(len(self.select_mask), meancoef[self.select_mask])
                # choose features with highest max values
                maxcoef = -np.amax(np.absolute(self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)]),axis = 0)
                self.select_mask = np.unique(np.append(self.select_mask, np.argsort(maxcoef)[:max_feat]))
                #print(len(self.select_mask), maxcoef[self.select_mask])
                # choose features with best ranking in conditions
                ranking = np.argsort(np.argsort(-np.absolute(self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)]),axis = 1), axis = 1)
                if (self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)] == 0).any():
                    ranking[self.coef_[:,:len(self.coef_[0])-int(self.fit_intercept)] == 0] = len(self.coef_[0]) - int(self.fit_intercept)
                minrank = np.amin(ranking,axis = 0)
                #print(minrank[self.select_mask])
                self.select_mask = np.unique(np.append(self.select_mask, np.argsort(minrank)[:max_feat]))
                # sort select mask by max values in any condition
                #print(self.select_mask)
                self.select_mask = self.select_mask[np.argsort(-np.amax(np.absolute(self.coef_[:,self.select_mask]),axis = 0))]
                print('Selected features', len(self.select_mask))
        else:
            self.select_mask = np.where(np.ones(len(self.coef_[0])-int(self.fit_intercept)) == 1)[0]
        return self.select_mask
    
    # computes the significance of a parameter being different from zero using bayesian posterior for parameters
    #https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    # https://gist.github.com/brentp/5355925
    def statistical_weight(self, X, Y, compute_new = True, comp_pvalues = True, logp = True, rcond = 1e-5, max_feat = 2000):
                
        if self.z_scores is None or compute_new:
            X = self.transform(X)
            yref = self.predict(X)
            
            if self.select_mask is None:
                self.reduced_coef(max_feat = max_feat)
            X = X[:,self.select_mask]
                        
            if self.fit_intercept:
                X = np.append(X, np.ones((len(X),1)), axis = 1)
            covar = np.dot(X.T, X)
            if self.penalty == 'l2':
                covar += np.eye(len(covar)) * self.alpha
                
            self.invcovardiag = np.linalg.pinv(covar, hermitian = True, rcond = rcond).diagonal()
            
            Mse = self.mse(yref, Y, axis = 0)/float(abs(self.shapeX[0]-self.shapeX[1]-int(self.fit_intercept)))
            
            var_b = Mse[:, None]*self.invcovardiag[None,:]
            if self.fit_intercept:
                var_b = var_b[:, :-1]
            self.sd_b = np.sqrt(var_b)
            
            self.z_scores = self.coef_[:,self.select_mask]/self.sd_b
            self.z_scores = np.nan_to_num(self.z_scores)
            
        if self.p_values is None and comp_pvalues:
            self.p_values = 2.*(1.-stats.t.cdf(np.absolute(self.z_scores),(abs(self.shapeX[0]-self.shapeX[1]-int(self.fit_intercept))) ))
            if logp:
                self.p_values[self.p_values == 0] = np.amin(self.p_values[self.p_values != 0])
                self.p_values = -np.log10(self.p_values)
                self.p_values = np.sign(self.z_scores)*np.absolute(self.p_values)
            return self.p_values, self.select_mask 
        return self.z_scores, self.select_mask





# define the predictor to be used        
class feature_predictor():
    def __init__(self, model, **params):
        self.model = model
        self.params = params
        
        if model == 'RandomForest' or model == 'RF' or model == 'randomforest':
            self.lr = ensemble.RandomForestRegressor(**params) #n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
        
        if model == 'LinearRegression' or model == 'OLS' or model == 'LR':
            self.lr = linear_model.LinearRegression(**params) #, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
        
        if model == 'ElasticNet' or model == 'elastic' or model == 'EN' or model == 'elasticnet':
            self.lr = linear_model.ElasticNet(**params) #alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        
    def fit(self, x, y):
        self.lr=self.lr.fit(x,y)
        self.x = x
        self.y = y
        
    def predict(self, x):
        pred = self.lr.predict(x)
        return pred
    
    def feature_importance(self):
        # assigns t-test statistic to LR coefficients, same as statsmodels.api.OLS
        if self.model == 'LinearRegression' or self.model == 'OLS' or self.model == 'LR' or self.model == 'ElasticNet' or self.model == 'elastic' or self.model == 'EN' or self.model == 'elasticnet':
            if self.lr.fit_intercept:
                params = np.append(self.lr.intercept_, self.lr.coef_)
                newX = np.append(np.ones((len(self.x),1)), self.x, axis=1)
            else:
                params = self.lr.coef_
                newX = self.x
            predictions = self.lr.predict(self.x)
            MSE = (sum((self.y-predictions)**2))/(len(newX)-len(newX[0]))
            var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params/sd_b
            #p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
            
            '''
            import statsmodels.api as sm
            X2 = sm.add_constant(self.x)
            est = sm.OLS(self.y, X2)
            est2 = est.fit()
            print(est2.summary())
            print(ts_b)
            print(self.lr.coef_)
            sys.exit()
            '''
            
            if self.lr.fit_intercept:
                ts_b = ts_b[1:]
            rank = np.argsort(np.argsort(-np.abs(ts_b))) + 1
            return rank
        
        elif self.model == 'RandomForest' or self.model == 'RF' or self.model == 'randomforest':
            ftimp = self.lr.feature_importances_
            return np.argsort(np.argsort(-ftimp)) + 1
    
    def coef_(self):
        if self.model == 'RandomForest' or self.model == 'RF' or self.model == 'randomforest':
            return self.lr.feature_importances_
        return self.lr.coef_
    
    def intercept_(self):
        if self.model == 'RandomForest' or self.model == 'RF' or self.model == 'randomforest':
            return 0
        return self.lr.intercept_







