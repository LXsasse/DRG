import sys, os 
import numpy as np
import scipy.stats as stats
from sklearn import linear_model, metrics
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from functools import reduce
from functions import correlation, mse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def print_averages(Y_pred, Ytest, testclasses, sysargv):
    if '--aurocaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'AUROC', metrics.roc_auc_score(Ytest[:,consider], Y_pred[:,consider]))
    if '--auprcaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'AUPRC', metrics.average_precision_score(Ytest[:,consider], Y_pred[:,consider]))
    if '--mseaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'MSE', mse(Ytest[:,consider], Y_pred[:,consider]), 'Var data', np.var(Ytest[:,consider]))
    if '--correlationaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'Correlation classes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 0)))
            print("Mean between classes", np.mean(cdist(Ytest[:,consider].T, Ytest[:,consider].T, 'correlation')[np.triu_indices(len(consider) ,1)]))
            print(tclass, 'Correlation genes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 1)))
    
def save_performance(Y_pred, Ytest, testclasses, experiments, names, outname, sysargv, compare_random = True):
    if '--save_correlation_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            correlations = correlation(Ytest[:,consider], Y_pred[:,consider], axis = 0)
            pvalue = 1.
            if compare_random:
                randomcorrelations = correlation(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 0)
                pvalue = stats.ttest_ind(correlations, randomcorrelations)[1]/2.
            np.savetxt(outname+'_exper_corr_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), correlations.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_auroc_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            auc = metrics.roc_auc_score(Ytest[:,consider], Y_pred[:,consider], average = None)
            pvalue = 1.
            if compare_random:
                randomauc = metrics.roc_auc_score(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], average = None)
                pvalue = stats.ttest_ind(auc, randomauc)[1]/2.
            np.savetxt(outname+'_exper_auroc_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), auc.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_auprc_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            auc = metrics.average_precision_score(Ytest[:,consider], Y_pred[:,consider], average = None)
            pvalue = 1.
            if compare_random:
                randomauc = metrics.average_precision_score(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], average = None)
                pvalue = stats.ttest_ind(auc, randomauc)[1]/2.
            np.savetxt(outname+'_exper_auprc_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), auc.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_mse_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            mses = mse(Ytest[:,consider], Y_pred[:,consider], axis = 0)
            pvalue = 1.
            if compare_random:
                randommses = mse(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 0)
                pvalue = stats.ttest_ind(mses, randommses)[1]/2.
            np.savetxt(outname+'_exper_mse_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    
    if '--save_correlation_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            correlations = correlation(Ytest[:,consider], Y_pred[:,consider], axis = 1)
            pvalue = 1.
            if compare_random:
                randomcorrelations = correlation(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 1)
                pvalue = stats.ttest_ind(correlations, randomcorrelations)[1]/2.
            np.savetxt(outname+'_gene_corr_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), correlations.reshape(-1,1),axis = 1), fmt = '%s', header ='P-value: '+str(pvalue))
    if '--save_mse_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            mses = mse(Ytest[:,consider], Y_pred[:,consider], axis = 1)
            pvalue = 1.
            if compare_random:
                randommses = mse(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 1)
                pvalue = stats.ttest_ind(mses, randommses)[1]/2.
            np.savetxt(outname+'_gene_mse_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_auroc_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            auc = metrics.roc_auc_score(Ytest[:,consider].T, Y_pred[:,consider].T, average = None)
            pvalue = 1.
            if compare_random:
                randomauc = metrics.roc_auc_score(Ytest[np.random.permutation(len(Ytest))][:,consider].T, Y_pred[:,consider].T, average = None)
                pvalue = stats.ttest_ind(auc, randomauc)[1]/2.
            np.savetxt(outname+'_gene_auroc_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), auc.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_auprc_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            auc = metrics.average_precision_score(Ytest[:,consider].T, Y_pred[:,consider].T, average = None)
            pvalue = 1.
            if compare_random:
                randomauc = metrics.average_precision_score(Ytest[np.random.permutation(len(Ytest))][:,consider].T, Y_pred[:,consider].T, average = None)
                pvalue = stats.ttest_ind(auc, randomauc)[1]/2.
            np.savetxt(outname+'_gene_auprc_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), auc.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_topdowncorrelation_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            topmask, botmask = Ytest >= 0, Ytest <= 0
            correlationstop = np.array([1.-pearsonr(Ytest[:, i][topmask[:,i]], Y_pred[:,i][topmask[:,i]])[0] for i in consider])
            correlationsbot = np.array([1.-pearsonr(Ytest[:, i][botmask[:,i]], Y_pred[:,i][botmask[:,i]])[0] for i in consider])
            print('Saved as', outname+'_exper_corrbottop_tcl'+tclass+'.txt')
            np.savetxt(outname+'_exper_corrbottop_tcl'+tclass+'.txt', np.concatenate([experiments[consider].reshape(-1,1), correlationstop.reshape(-1,1),correlationsbot.reshape(-1,1)],axis = 1), fmt = '%s')

def plot_scatter(Ytest, Ypred, titles = None, xlabel = None, ylabel = None, outname = None, include_lr = True, include_mainvar = True):
    n = len(Ytest[0])
    if n > 100:
        print('Number of examples is too large', n)
        return
    x_col = int(np.sqrt(n))
    y_row = int(n/x_col) + int(n%x_col!= 0)
    fig = plt.figure(figsize = (x_col*1.5,y_row*1.5), dpi = 100)
    for e in range(n):
        ax = fig.add_subplot(y_row, x_col, e+1)
        pcorr = pearsonr(Ytest[:,e], Ypred[:,e])[0]
        if titles is not None:
            ax.set_title(titles[e]+' R='+str(np.around(pcorr,2)), fontsize = 6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.scatter(Ytest[:,e], Ypred[:,e], c = 'slategrey', alpha = 0.7, s = 6)
        limx, limy = ax.get_xlim(), ax.get_ylim()
        lim = [max(limx[0], limy[0]), min(limx[1], limy[1])]
        ax.plot(lim, lim, color = 'maroon', ls = '--')
        if include_lr:
            lr = linear_model.LinearRegression().fit(Ytest[:, [e]], Ypred[:,e])
            ax.plot(np.array(limx), lr.predict(np.array(limx).reshape(-1,1)), color = 'k')
        if include_mainvar:
            centerx, centery = np.mean(Ytest[:,e]), np.mean(Ypred[:,e])
            maindir, u, v = np.linalg.svd(np.array([Ytest[:,e]-centerx, Ypred[:,e]-centery]), full_matrices=False)
            maindir = maindir[:,0]
            slope = maindir[1]/maindir[0]
            bias = centery-slope*centerx
            ax.plot(np.array(limx), np.array(limx)*slope + bias, color = 'k')
        
    
    if xlabel is not None:
        fig.text(0.5, 0.05-0.25/y_row, xlabel, ha='center')
    if ylabel is not None:
        fig.text(0.05-0.2/x_col, 0.5, ylabel, va='center', rotation='vertical')
    if outname is not None:
        print('SAVED as', outname)
        fig.savefig(outname, dpi = 200, bbox_inches = 'tight')
    else:
        fig.tight_layout()
        plt.show()
       
    

    
