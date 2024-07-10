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















    

if __name__ == '__main__':
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    
    
    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = str(sys.argv[sys.argv.index('--delimiter')+1])
        print(delimiter)
    X, Y, names, features, experiments = readin(inputfile, outputfile, delimiter = delimiter,return_header = True)
    
    if ',' in inputfile:
        inputfiles = inputfile.split(',')
        inputfile = inputfiles[0]
        for inp in inputfiles[1:]:
            inputfile = create_outname(inp, inputfile, lword = 'and')
            print(inputfile)
    outname = create_outname(inputfile, outputfile) 
    print(outname)
    
    
    if '--outdir' in sys.argv:
        outname = sys.argv[sys.argv.index('--outdir')+1] + os.path.split(outname)[1]

    
    X, features = manipulate_input(X, features, sys.argv)
    
    # use a random number of samples from the entire data set to test model
    if '--testrandom' in sys.argv:
        subsamp = float(sys.argv[sys.argv.index('--testrandom')+1])
        if subsamp < 1:
            subsamp = int(len(X)*subsamp)
        subsamp = int(subsamp)
        outname +='ss'+str(subsamp)
        sub = np.random.permutation(len(X))[:subsamp]
        names, X, Y = names[sub], X[sub], Y[sub]
    
    if '--crossvalidation' in sys.argv:
        folds = check(sys.argv[sys.argv.index('--crossvalidation')+1])
        fold = int(sys.argv[sys.argv.index('--crossvalidation')+2])
        if '--significant_genes' in sys.argv:
            siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
            Yclass = np.isin(names, siggenes).astype(int)
        elif '--gene_classes' in sys.argv:
            Yclass = []
            for n, name in enumerate(names):
                Yclass.append(name.split(':')[0])
            Yclass = np.array(Yclass)
        else:    
            cutoff = float(sys.argv[sys.argv.index('--crossvalidation')+3])
            Yclass = (np.sum(np.absolute(Y)>=cutoff, axis = 1) > 0).astype(int)
    else:
        folds, fold, Yclass = 10, 0, None
        
    if isinstance(folds,int):
        outname += '-cv'+str(folds)+'-'+str(fold)
    else:
        outname += '-cv'+str(int(cutoff))+'-'+str(fold)
    trainset, testset, valset = create_sets(len(X), folds, fold, Yclass = Yclass, genenames = names)
    
    print('Train', len(trainset))
    print('Test', len(testset), testset)
    print('Val', len(valset))
    
    
    if '--norm2output' in sys.argv:
        print ('ATTENTION: output has been normalized along data points')
        outname += '-n2out'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 1))[:, None] 
        Y = Y/outnorm
        
    elif '--norm2outputclass' in sys.argv:
        print('ATTENTION: output has been normalized along data classess')
        outname += '-n2outclass'
        outnorm =np.sqrt(np.sum(Y*Y, axis = 0))
        Y = Y/outnorm

    weights = None
    if '--sample_weights' in sys.argv:
        weightfile = np.genfromtxt(sys.argv[sys.argv.index('--sample_weights')+1], dtype = str)
        outname += '_sweigh'
        weight_names, weights = weightfile[:,0], weightfile[:,1].astype(float)
        sortweight = np.argsort(weight_names)[np.isin(np.sort(weight_names), names)]
        weight_names, weights = weight_names[sortweight], weights[sortweight]
        if not np.array_equal(weight_names, names):
            print("Weights cannot be used")
            sys.exit()
        weights = weights[trainset]
    
    #alpha=1.0, fit_intercept=True, max_iter=None, tol=None, solver='auto', positive=False, random_state=None, penalty = None, pca = None, center = False, optimize_alpha = True, change_alpha =.6, validation_axis = 1, alpha_search = 'independent', normalize = False, full_nonlinear = False, logistic = False, warm_start = False, n_jobs = None
    param_names = {'alpha':'l1reg_last', 'fit_intercept':'kernel_bias', 'penalty':'loss_function', 'max_iter':'epochs', 'tol':'patience', 'solver':'optimizer', 'positive':'kernel_function', 'random_state':'seed', 'pca': 'num_kernels', 'center': 'center', 'optimize_alpha': 'init_adjust', 'change_alpha':'kernel_lr', 'validation_axis': 'optim_params', 'alpha_search': 'adjust_lr', 'normalize': 'batch_norm', 'full_nonlinear': 'interaction_layer', 'logistic': 'outclass', 'warm_start': 'hot_start', 'n_jobs': 'device'}
    
    if '--skregression' in sys.argv:
        fit_intercept = check(sys.argv[sys.argv.index('--skregression')+1])
        penalty = sys.argv[sys.argv.index('--skregression')+2]
        alpha = float(sys.argv[sys.argv.index('--skregression')+3])

        outname += '_reglr'+penalty+'-'+str(alpha)+'fi'+str(fit_intercept)[0]
        
        params = {}
        if len(sys.argv) > sys.argv.index('--skregression')+4:
            if '--' not in sys.argv[sys.argv.index('--skregression')+4]:
                if '+' in sys.argv[sys.argv.index('--skregression')+4]:
                    parameters = sys.argv[sys.argv.index('--skregression')+4].split('+')
                else:
                    parameters = [sys.argv[sys.argv.index('--skregression')+4]]
                for p in parameters:
                    if ':' in p and '=' in p:
                        p = p.split('=',1)
                    elif ':' in p:
                        p = p.split(':',1)
                    elif '=' in p:
                        p = p.split('=',1)
                    params[p[0]] = check(p[1])
                    outname += p[0][:2]+p[0][max(2,len(p[0])-2):]+str(p[1])
        
        model = sk_regression(fit_intercept = fit_intercept, alpha = alpha, penalty = penalty, **params)
        
        obj= open(outname+'_model_params.dat', 'w')
        obj.write('outname : '+outname+'\n'+'kernel_bias : '+str(fit_intercept)+'\n'+'loss_function : '+str(penalty)+'\n'+'l1reg_last : '+str(alpha)+'\n')
        if len(params) > 0:
            for par in params:
                obj.write(param_names[par] + ' : '+str(params[par])+'\n')
                
        print(outname+'_model_params.dat')
        
        
    print('Fit')
    model.fit(X[trainset], Y[trainset], XYval = [X[valset], Y[valset]], weights = weights)
    Y_pred = model.predict(X[testset])
    
    if '--norm2output' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm[testset]
        
    elif '--norm2outputclass' in sys.argv:
        Y *= outnorm
        Y_pred *= outnorm
    
    if '--random_model' in sys.argv:
        could_add_modelthatlearns_on_random_data = 0
    
    if '--save_predictions' in sys.argv:
        np.savetxt(outname+'_pred.txt', np.append(names[testset][:, None], Y_pred, axis = 1), header = ' '.join(experiments), fmt = '%s')
        print('SAVED', outname+'_pred.txt')    

    if '--split_outclasses' in sys.argv:
        testclasses = np.genfromtxt(sys.argv[sys.argv.index('--split_outclasses')+1], dtype = str)
        tsort = []
        for exp in experiments:
            tsort.append(list(testclasses[:,0]).index(exp))
        testclasses = testclasses[tsort][:,1]
    else:
        testclasses = np.zeros(len(Y[0]), dtype = np.int8).astype(str)
    
    if '--significant_genes' in sys.argv:
        siggenes = np.genfromtxt(sys.argv[sys.argv.index('--significant_genes')+1], dtype = str)
        tlen = len(testset)
        tsetmask = np.isin(names[testset], siggenes)
        testset = testset[tsetmask]
        Y_pred = Y_pred[tsetmask]
        print('Testset length reduced to significant genes from', tlen, 'to', len(testset))

    # USE: --aurocaverage --auprcaverage --mseaverage --correlationaverage
    print_averages(Y_pred, Y[testset], testclasses, sys.argv)
    
    # USE: --save_correlation_perclass --save_auroc_perclass --save_auprc_perclass --save_mse_perclass --save_correlation_pergene '--save_mse_pergene --save_auroc_pergene --save_auprc_pergene --save_topdowncorrelation_perclass
    save_performance(Y_pred, Y[testset], testclasses, experiments, names[testset], outname, sys.argv, compare_random = True)
    
    # reducing X to features that
    features = np.array(features)
    if model.Xmask is not None:
        X, features = X[:, model.Xmask], np.array(features)[model.Xmask]
        model.Xmask = None
    
    if '--feature_weights' in sys.argv:
        coef_mask = model.reduced_coef()
        np.savetxt(outname+'_features.dat', np.append(features[coef_mask].reshape(-1,1),np.around(model.coef_.T[coef_mask],5), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_zscores' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues, coef_mask = model.statistical_weight(X[trainset], Y[trainset], comp_pvalues = False)
        np.savetxt(outname+'_feature_zscores.dat', np.append(features[coef_mask].reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
    if '--feature_statistics' in sys.argv:
        # saves sign(coef)*log10(pvalues)
        logpvalues, coef_mask = model.statistical_weight(X[trainset], Y[trainset], compute_new = False)
        np.savetxt(outname+'_feature_stats.dat', np.append(features[coef_mask].reshape(-1,1),np.around(logpvalues.T,2), axis = 1).astype(str), fmt = '%s', header = ' '.join(experiments))
    
        
        # plots scatter plot for each output class
    if '--plot_correlation_perclass' in sys.argv:
        plot_scatter(Y[testset], Y_pred, xlabel = 'Measured', ylabel = 'Predicted', titles = experiments, outname = outname + '_class_scatter.jpg')
        
    # plots scatter plot fo n_genes that within the n_genes quartile
    if '--plot_correlation_pergene' in sys.argv:
        n_genes = int(sys.argv['--plot_correlation_pergene'])
        for tclass in np.unique(testclasses):
            correlation_genes = correlation(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 1)
            sort = np.argsort(correlation_genes)
            posi = np.linspace(0,len(correlation_genes), n_genes).astype(int)
            i = sort[posi] 
            plot_scatter(Y[test][i].T, Ypred[i].T, xlabel = 'Measured', ylabel = 'Predicted', titles = names[testset][i], outname = outname + '_gene_scatter.jpg')








