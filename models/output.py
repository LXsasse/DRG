import sys, os 
import numpy as np
import scipy.stats as stats
from sklearn import linear_model, metrics
from scipy.stats import pearsonr, ranksums
from scipy.spatial.distance import cdist, jensenshannon
from functools import reduce
from functions import correlation, mse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def avgpool(x,window):
    lx = np.shape(x)[-1]
    if lx%window!=0:
        xtend = [int(np.floor((lx%window)/2)), int(np.ceil((lx%window)/2))]
        x = np.pad(x, pad_width = [[0,0],[0,0],xtend])
    lx = np.shape(x)[-1]
    xavg = np.zeros(list(np.shape(x)[:-1])+[int(lx/window)])
    for i in range(int(lx/window)):
        xavg[..., i] = np.mean(x[..., i*window:(i+1)*window], axis = -1)
    return xavg
    

def print_averages(Y_pred, Ytest, testclasses, sysargv):
    if '--aurocaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'AUROC', metrics.roc_auc_score(Ytest[:,consider], Y_pred[:,consider]))
    if '--auprcaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print("Expected", np.mean(Ytest[:,consider], axis = 0))
            print(tclass, 'AUPRC', metrics.average_precision_score(Ytest[:,consider], Y_pred[:,consider]))
    if '--mseaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'MSE', mse(Ytest[:,consider], Y_pred[:,consider]))
    if '--correlationaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, '1-Correlation classes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 0)))
            print("1-Correlation Mean between classes", np.mean(cdist(Ytest[:,consider].T, Ytest[:,consider].T, 'correlation')[np.triu_indices(len(consider) ,1)]))
            print("1-Correlation to Mean for classes", np.mean(correlation(Ytest[:,consider].T, np.array(len(consider)*[np.mean(Ytest[:,consider],axis =1)]), axis = 1)))
            print(tclass, '1-Correlation genes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 1)))
    if '--msemeanaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(np.shape(np.sum(Ytest[:,consider], axis = -1)), np.shape(np.sum(Y_pred[:,consider],axis = -1)))
            print(tclass, 'MSEMEAN', mse(np.sum(Ytest[:,consider], axis = -1), np.sum(Y_pred[:,consider],axis = -1)))
    if '--jensenshannonaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Y_predc = np.copy(Y_pred)
            Y_predc[Y_predc < 0] = 0
            print(tclass, 'JShannon', np.mean(jensenshannon(np.transpose(Ytest[:,consider]), np.transpose(Y_predc[:,consider]))))#np.mean(jensenshannon(Ytest[:,consider], Y_pred[:,consider], axis = -1))) available as soon as conda updates to version 1.7
    if '--wilcoxonaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'Wilcoxon', np.mean(np.absolute(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[0])))
    if '--wilcoxonpaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'Wilcoxon', np.mean(-np.log10(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[1])))
            
        
    
    
def save_performance(Y_pred, Ytest, testclasses, experiments, names, outname, sysargv, compare_random = True):
    if '--save_msemean_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            mses = mse(np.sum(Ytest[:,consider], axis = -1), np.sum(Y_pred[:,consider],axis = -1),axis = 0)
            np.savetxt(outname+'_exper_msemean_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_jensenshannon_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Y_predc = np.copy(Y_pred)
            Y_predc[Y_predc < 0] = 1e-8
            JShannon = np.mean(np.transpose(jensenshannon(np.transpose(Ytest[:,consider]+1e-8), np.transpose(Y_predc[:,consider]))), axis = 0)
            np.savetxt(outname+'_exper_js_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), JShannon.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_wilcoxon_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Wilcoxon = np.mean(np.absolute(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[0]), axis = 0)
            np.savetxt(outname+'_exper_wlcx_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), Wilcoxon.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_wilcoxonp_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Wilcoxon = np.mean(-np.log10(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[1]), axis = 0)
            np.savetxt(outname+'_exper_wlcxp_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), Wilcoxon.reshape(-1,1),axis = 1), fmt = '%s')
    
    if '--save_correlation_perclass' in sysargv:
        Ytes = np.copy(Ytest)
        Y_pre = np.copy(Y_pred)
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            if len(np.shape(Ytes))>2:
                Ytes = np.sum(Ytes, axis = -1)
                Y_pre = np.sum(Y_pre, axis = -1)
            correlations = correlation(Ytes[:,consider], Y_pre[:,consider], axis = 0)
            pvalue = 1.
            if compare_random:
                randomcorrelations = correlation(Ytes[np.random.permutation(len(Ytes))][:,consider], Y_pre[:,consider], axis = 0)
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
            axis = 0
            if len(np.shape(Ytest))> 2:
                axis = (0,-1)
            mses = mse(Ytest[:,consider], Y_pred[:,consider], axis = axis)
            pvalue = 1.
            if compare_random:
                randommses = mse(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = 0)
                pvalue = stats.ttest_ind(mses, randommses)[1]/2.
            np.savetxt(outname+'_exper_mse_tcl'+tclass+'.txt', np.append(experiments[consider].reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s', header = 'P-value: '+str(pvalue))
    if '--save_topdowncorrelation_perclass' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            topmask, botmask = Ytest >= 0, Ytest <= 0
            correlationstop = np.array([1.-pearsonr(Ytest[:, i][topmask[:,i]], Y_pred[:,i][topmask[:,i]])[0] for i in consider])
            correlationsbot = np.array([1.-pearsonr(Ytest[:, i][botmask[:,i]], Y_pred[:,i][botmask[:,i]])[0] for i in consider])
            print('Saved as', outname+'_exper_corrbottop_tcl'+tclass+'.txt')
            np.savetxt(outname+'_exper_corrbottop_tcl'+tclass+'.txt', np.concatenate([experiments[consider].reshape(-1,1), correlationstop.reshape(-1,1),correlationsbot.reshape(-1,1)],axis = 1), fmt = '%s')
    
    
    if '--save_correlation_pergene' in sysargv or '--save_correlationmean_pergene' in sysargv:
        Ytes = np.copy(Ytest)
        Y_pre = np.copy(Y_pred)
        for tclass in np.unique(testclasses):
            if '--save_correlationmean_pergene' in sysargv:
                Ytes = np.sum(Ytes, axis = -1)
                Y_pre = np.sum(Y_pre, axis = -1)
            consider = np.where(testclasses == tclass)[0]
            correlations = correlation(Ytes[:,consider], Y_pre[:,consider], axis = -1)
            pvalue = 1.
            if compare_random:
                randomcorrelations = correlation(Ytes[np.random.permutation(len(Ytes))][:,consider], Y_pre[:,consider], axis = -1)
                pvalue = stats.ttest_ind(correlations, randomcorrelations)[1]/2.
            np.savetxt(outname+'_gene_corr_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), correlations.reshape(-1,1),axis = 1), fmt = '%s', header ='P-value: '+str(pvalue))
    if '--save_mse_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            axis = -1
            if len(np.shape(Ytest))>2:
                axis = (1,-1)
            mses = mse(Ytest[:,consider], Y_pred[:,consider], axis = axis)
            pvalue = 1.
            if compare_random:
                randommses = mse(Ytest[np.random.permutation(len(Ytest))][:,consider], Y_pred[:,consider], axis = -1)
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
    
    if '--save_msewindow_pergene' in sys.argv:
        windowsize = 10
        Ytes = np.copy(Ytest)
        Y_pre = np.copy(Y_pred)
        Ytes = avgpool(Ytes, windowsize)
        Y_pre = avgpool(Y_pre, windowsize)
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            mses = mse(np.sum(Ytes[:,consider], axis = -1), np.sum(Y_pre[:,consider],axis = -1),axis = 1)
            np.savetxt(outname+'_gene_msewindow_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s')

    if '--save_msemean_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            mses = mse(np.sum(Ytest[:,consider], axis = -1), np.sum(Y_pred[:,consider],axis = -1),axis = 1)
            np.savetxt(outname+'_gene_msemean_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), mses.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_jensenshannon_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Y_predc = np.copy(Y_pred)
            Y_predc[Y_predc < 1e-8] = 1e-8
            JShannon = np.mean(np.transpose(jensenshannon(np.transpose(Ytest[:,consider]+1e-8), np.transpose(Y_predc[:,consider]))), axis = 1)
            np.savetxt(outname+'_gene_js_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), JShannon.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_wilcoxon_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Wilcoxon = np.mean(np.absolute(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[0]), axis = 1)
            np.savetxt(outname+'_gene_wlcx_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), Wilcoxon.reshape(-1,1),axis = 1), fmt = '%s')
    if '--save_wilcoxonp_pergene' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            Wilcoxon = np.mean(-np.log10(ranksums(Ytest[:,consider], Y_pred[:,consider], axis = -1)[1]), axis = 1)
            np.savetxt(outname+'_gene_wlcxp_tcl'+tclass+'.txt', np.append(names.reshape(-1,1), Wilcoxon.reshape(-1,1),axis = 1), fmt = '%s')





def plot_scatter(Ytest, Ypred, titles = None, xlabel = None, ylabel = None, indsize = 3.5, dpi = 300, outname = None, include_lr = False, include_mainvar = False, dotlabel = None):
    n = len(Ytest)
    if n > 100:
        print('Number of examples is too large', n)
        return
    y_row = int(np.sqrt(n))
    x_col = int(n/y_row) + int(n%y_row!= 0)
    fig = plt.figure(figsize = ((x_col+0.3)*indsize,y_row*indsize), dpi = 100)
    for e in range(n):
        ax = fig.add_subplot(y_row, x_col, e+1)
        
        pcorr = pearsonr(Ytest[e], Ypred[e])[0]
        if titles is not None:
            ax.set_title(titles[e], fontsize = int(indsize*3))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        
        ax.scatter(Ytest[e], Ypred[e], c = 'slategrey', alpha = 0.7, s = int(indsize*10), label = ' R='+str(np.around(pcorr,2)))
        
        limx, limy = ax.get_xlim(), ax.get_ylim()
        lim = [max(limx[0], limy[0]), min(limx[1], limy[1])]
        ax.plot(lim, lim, color = 'maroon', ls = '--')
        
        if include_lr:
            lr = linear_model.LinearRegression().fit(Ytest[e], Ypred[e])
            ax.plot(np.array(limx), lr.predict(np.array(limx).reshape(-1,1)), color = 'k')
        if include_mainvar:
            centerx, centery = np.mean(Ytest[e]), np.mean(Ypred[e])
            maindir, u, v = np.linalg.svd(np.array([Ytest[e]-centerx, Ypred[e]-centery]), full_matrices=False)
            maindir = maindir[:,0]
            slope = maindir[1]/maindir[0]
            bias = centery-slope*centerx
            ax.plot(np.array(limx), np.array(limx)*slope + bias, color = 'k')
    
        ax.legend(fontsize = int(indsize*3))
        if dotlabel is not None:
            for d, dl in enumerate(dotlabel):
                ax.text(Ytest[e][d], Ypred[e][d], dl, ha = 'left', va = 'bottom', fontsize = int(indsize*3))
        
    if xlabel is not None:
        fig.text(0.5, 0.07/y_row, xlabel, ha='center')
    if ylabel is not None:
        fig.text(0.07, 0.5, ylabel, va='center', rotation='vertical')
    fig.subplots_adjust(wspace=0.5,hspace=0.5)
    if outname is not None:
        print('SAVED as', outname)
        fig.savefig(outname, dpi = dpi, bbox_inches = 'tight')
    else:
        fig.tight_layout()
        plt.show()
 
 
 
 
 
 
def unique_ordered(alist):
    a = []
    for l in alist:
        if l not in a:
            a.append(l)
    return a
        
def add_params_to_outname(outname, ndict):
    
    if isinstance(ndict['loss_function'], list):
        lssf = ''
        for lsf in unique_ordered(ndict['loss_function']):
            lssf += lsf[:2]+lsf[max(2,len(lsf)-2):]
    else:
        lssf = ndict['loss_function'][:3]+ndict['loss_function'][max(3,len(ndict['loss_function'])-3):]
    
    outname += '_'+lssf+'k'+str(ndict['num_kernels'])+'l'+str(ndict['l_kernels'])+str(ndict['kernel_bias'])[0]+'f'+ndict['kernel_function']
    
    if 'nlconv' in ndict:
        if ndict['nlconv']:
            outname+='NL'+str(ndict['nlconv_nfc'])
            if ndict['nlconv_position_wise']:
                outname += 'P'
            if ndict['nlconv_explicit']:
                outname += 'E'
    
    if ndict['net_function'] != ndict['kernel_function']:
        outname += ndict['net_function']
    
    if ndict['max_pooling'] and ndict['mean_pooling']:
        outname +='mmpol'+str(ndict['pooling_size'])[:3]
        if ndict['pooling_steps'] != ndict['pooling_size']:
            outname += 'st'+str(ndict['pooling_steps'])[:3]
        if ndict['kernel_thresholding'] != 0:
            outname += 'kt'+str(ndict['kernel_thresholding'])
            
    elif ndict['max_pooling']:
        outname +='max'+str(ndict['pooling_size'])[:3]
        if ndict['pooling_steps'] != ndict['pooling_size']:
            outname += 'st'+str(ndict['pooling_steps'])[:3]
        if ndict['kernel_thresholding'] != 0:
            outname += 'kt'+str(ndict['kernel_thresholding'])
    elif ndict['mean_pooling']:
        outname +='mean'+str(ndict['pooling_size'])[:3]
        if ndict['pooling_steps'] != ndict['pooling_size']:
            outname += 'st'+str(ndict['pooling_steps'])[:3]
        if ndict['kernel_thresholding'] != 0:
            outname += 'kt'+str(ndict['kernel_thresholding'])
    elif ndict['weighted_pooling']:
        outname +='wei'+str(ndict['pooling_size'])[:3]
        if ndict['pooling_steps'] != ndict['pooling_size']:
            outname += 'st'+str(ndict['pooling_steps'])[:3]
        if ndict['kernel_thresholding'] != 0:
            outname += 'kt'+str(ndict['kernel_thresholding'])
       
    if 'reverse_complement' in ndict:
        if ndict['reverse_complement']:
            outname += 'rcT'
    
    if 'l_out' in ndict:
        if ndict['l_out'] is None:
            outname += 'bp1'
        outname += 'bp'+str(int(ndict['l_seqs']/ndict['l_out']))
    
    if ndict['validation_loss'] != ndict['loss_function'] and ndict['validation_loss'] is not None:
        if isinstance(ndict['validation_loss'], list):
            lssf = ''
            for lsf in unique_ordered(ndict['validation_loss']):
                lssf += lsf[:2]+lsf[max(2,len(lsf)-2):]
        else:
            lssf = str(ndict['validation_loss'])[:2]+str(ndict['validation_loss'])[max(2,len(str(ndict['validation_loss']))-2):]
        outname +='vl'+lssf
    if ndict['hot_start']:
        outname += '-hot'
    if ndict['warm_start']:
        outname += '-warm'
    if ndict['shift_sequence'] is not None:
        if isinstance(ndict['shift_sequence'], int):
            if ndict['shift_sequence'] > 0:
                outname += 'sft'+str(ndict['shift_sequence'])
        else:
            outname += 'sft'+str(np.amax(ndict['shift_sequence']))
        if ndict['random_shift']:
            outname+=str(int(ndict['random_shift']))
    if ndict['smooth_onehot']:
        outname += 'smo'
    
    if ndict['reverse_sign']:
        outname += 'rs'
    
    
    if ndict['restart']:
        outname += 're'
   
    if ndict['gapped_convs'] is not None:
        if len(ndict['gapped_convs']) > 0:
            outname + '_gapc'
            glist = ['k','g','n','s']
            for gl in range(len(ndict['gapped_convs'])):
                for g in range(4):
                    if gl == 0 or ndict['gapped_convs'][gl][g] != ndict['gapped_convs'][max(0,gl-1)][g]:
                        outname += glist[g]+str(ndict['gapped_convs'][gl][g])
        if ndict['gapconv_residual']:
            outname += 'T'
        if ndict['gapconv_pooling']:
            outname += 'T'
        
        if 'final_convolutions' in ndict:
            if ndict['final_convolutions'] > 0:
                outname += 'fcnv'+str(ndict['final_convolutions'])+'l'+str(ndict['l_finalkernels'])
                if ndict['final_conv_dim'] is not None:
                    outname += 'd'+str(ndict['final_conv_dim'])
                if ndict['finalstrides'] != 1:
                    outname += 's'+str(ndict['finalstrides'])
                if ndict['finaldilations'] != 1:
                    outname += 'i'+str(ndict['finaldilations'])
                
                
        if 'finalmax_pooling' in ndict:
            if ndict['finalmax_pooling'] > 0:
                outname += 'fmap'+str(ndict['finalmax_pooling'])
        if 'finalmeanpooling' in ndict:
            if ndict['finalmean_pooling'] > 0:
                outname += 'fmep'+str(ndict['finalmean_pooling'])
        if 'finalweighted_pooling' in ndict:
            if ndict['finalweighted_pooling'] > 0:
                outname += 'fwei'+str(ndict['finalweighted_pooling'])
        
    if ndict['dilated_convolutions'] > 0:
        outname += '_dc'+str(ndict['dilated_convolutions'])+'i'+str(ndict['conv_increase']).strip('0').strip('.')+'d'+str(ndict['dilations']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(ndict['strides']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','') +'l'+str(ndict['l_dilkernels'])
        if ndict['dilmax_pooling'] > 0:
            outname += 'da'+str(ndict['dilmax_pooling'])
            if ndict['dilpooling_steps'] != 1:
                outname += 'st'+str(ndict['dilpooling_steps'])                
        if ndict['dilmean_pooling'] > 0:
            outname += 'de'+str(ndict['dilmean_pooling'])
            if ndict['dilpooling_steps'] != 1:
                outname += 'st'+str(ndict['dilpooling_steps'])
        if ndict['dilweighted_pooling'] > 0:
            outname += 'dw'+str(ndict['dilweighted_pooling'])
            if ndict['dilpooling_steps'] != 1:
                outname += 'st'+str(ndict['dilpooling_steps'])
        if ndict['dilpooling_residual'] > 0:
            outname += 'r'+str(ndict['dilpooling_residual'])
        if ndict['dilresidual_entire'] > 0:
            outname += 're'
        if ndict['dilresidual_concat']:
            outname += 'ccT'
    
    if ndict['embedding_convs'] > 0:
        outname += 'ec'+str(ndict['embedding_convs'])
    
    if ndict['n_transformer'] >0:
        outname += 'trf'+str(ndict['sum_attention'])[0]+str(ndict['n_transformer'])+'h'+str(ndict['n_distattention'])+'-'+str(ndict['dim_distattention'])
    
    elif ndict['n_interpolated_conv'] > 0:
        outname += 'iplcv'+str(ndict['n_interpolated_conv'])+'-'+str(ndict['dim_embattention'])+'-'+str(ndict['dim_distattention'])
        if ndict['n_distattention'] is not None:
            outname+='-'+str(ndict['n_distattention'])
        if ndict['sum_attention']:
            outname += 'sa'
        if ndict['attentionconv_pooling'] > 1:
            outname += 'mc'+str(ndict['attentionconv_pooling'])
        if ndict['attentionmax_pooling'] > 0:
            outname += 'ma'+str(ndict['attentionmax_pooling'])
        if ndict['attentionweighted_pooling'] > 0:
            outname += 'mw'+str(ndict['attentionweighted_pooling']) 
    
    elif ndict['n_attention'] > 0:
        outname += 'at'+str(ndict['n_attention'])+'h'+str(ndict['n_distattention'])+'-'+str(ndict['dim_distattention'])
        
        if ndict['dim_embattention'] is not None:
            outname += 'v'+str(ndict['dim_embattention'])
        if ndict['attentionmax_pooling'] > 0:
            outname += 'ma'+str(ndict['attentionmax_pooling'])
        if ndict['attentionweighted_pooling'] > 0:
            outname += 'mw'+str(ndict['attentionweighted_pooling'])            
    
    elif ndict['n_hyenaconv'] > 0:
        outname += 'hyna'+str(ndict['n_hyenaconv'])+str(ndict['n_distattention'])
        
        if ndict['dim_embattention'] is not None:
            outname += 'v'+str(ndict['dim_embattention'])
        if ndict['attentionmax_pooling'] > 0:
            outname += 'ma'+str(ndict['attentionmax_pooling'])
        if ndict['attentionweighted_pooling'] > 0:
            outname += 'mw'+str(ndict['attentionweighted_pooling']) 
    
    
    if ndict['transformer_convolutions'] > 0:
        outname += '_tc'+str(ndict['transformer_convolutions'])+'d'+str(ndict['trconv_dim'])+'d'+str(ndict['trdilations']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(ndict['trstrides']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')
        
        if ndict['trpooling_residual'] > 0:
            outname += 'r'+str(ndict['trpooling_residual'])
        if ndict['trresidual_entire'] > 0:
            outname += 're'
      
        outname+='l'+str(ndict['l_trkernels'])
        if ndict['trmax_pooling']>0:
            outname += 'ma'+str(ndict['trmax_pooling'])
            if ndict['trpooling_steps'] != 1:
                outname += 'st'+str(ndict['trpooling_steps'])
        if ndict['trmean_pooling']>0:
            outname += 'me'+str(ndict['trmean_pooling'])
            if ndict['trpooling_steps'] != 1:
                outname += 'st'+str(ndict['trpooling_steps'])
        if ndict['trweighted_pooling']>0:
            outname += 'mw'+str(ndict['trweighted_pooling'])
            if ndict['trpooling_steps'] != 1:
                outname += 'st'+str(ndict['trpooling_steps'])
        
    nfcgiven = False
    if 'nfc_layers' in ndict:
        if isinstance(ndict['nfc_layers'], list):
            if len(np.unique(ndict['nfc_layers'])) == 1:
                nfc = str(ndict['nfc_layers'][0])
            else:
                nfc = ''.join(np.array(ndict['nfc_layers']).astype(str))
            outname +='nfc'+nfc 
            nfcgiven = True
        elif ndict['nfc_layers'] > 0:
            outname +='nfc'+str(ndict['nfc_layers'])
            nfcgiven = True
    if ndict['fc_function'] != ndict['net_function'] and nfcgiven:
        outname += ndict['fc_function']
    if ndict['fclayer_size'] is not None and nfcgiven:
        outname += 's'+str(ndict['fclayer_size'])
        
        
    if nfcgiven and ndict['nfc_residuals'] > 0:
        outname += 'r'+str(ndict['nfc_residuals'])
        
        
    if 'interaction_layer' in ndict:
        if ndict['interaction_layer']:
            outname += '_intl'+str(ndict['interaction_layer'])[0]
    elif 'neuralnetout' in ndict:
        if ndict['neuralnetout'] > 0:
            outname += 'nno'+str(ndict['neuralnetout'])
    if 'final_kernel' in ndict:
        outname += str(ndict['final_kernel'])+'-'+str(ndict['final_strides'])+str(ndict['predict_from_dist'])[0]
    
    
    if 'outclass' in ndict:
        if isinstance(ndict['outclass'], list):
            if len(np.unique(ndict['outclass'])) == 1:
                outname += ndict['outclass'][0][:2]+ndict['outclass'][0][max(2,len(ndict['outclass'][0])-2):]
            else:
                for otcl in ndict['outclass']:
                    outname += otcl[:2]+otcl[max(2,len(otcl)-2):]
        elif ndict['outclass'] != 'Linear':
            outname += ndict['outclass'][:2]+ndict['outclass'][-max(2,len(ndict['outclass'])-2):]
        if ndict['outclass'] == 'LOGXPLUSFRACTION': 
            outname += str(ndict['outlog']) + str(ndict['outlogoffset'])
            
    
    if ndict['l1reg_last'] > 0:
        outname += 'l1'+str(ndict['l1reg_last'])
    if ndict['l2reg_last'] > 0:
        outname += 'l2'+str(ndict['l2reg_last'])
    if ndict['l1_kernel'] > 0:
        outname += 'l1k'+str(ndict['l2reg_last'])
    if ndict['dropout'] > 0.:
        outname += 'do'+str(ndict['dropout'])
    if ndict['batch_norm']:
        outname += 'bno'+str(ndict['batch_norm'])[0]
    
    if ndict['conv_dropout'] > 0.:
        outname += 'cdo'+str(ndict['conv_dropout'])
    if ndict['conv_batch_norm']:
        outname += 'cbno'+str(ndict['conv_batch_norm'])[0]
    if ndict['attention_dropout'] > 0.:
        outname += 'ado'+str(ndict['attention_dropout'])
    if ndict['attention_batch_norm']:
        outname += 'abno'+str(ndict['attention_batch_norm'])[0]
    if ndict['fc_dropout'] > 0.:
        outname += 'fdo'+str(ndict['fc_dropout'])
    if ndict['fc_batch_norm']:
        outname += 'fbno'+str(ndict['fc_batch_norm'])[0]
    
    outname += 'tr'+str(ndict['lr'])+ndict['optimizer']
    
    if ndict['optim_params'] is not None:
        outname += str(ndict['optim_params']).replace('(', '-').replace(')', '').replace(',', '-').replace(' ', '')
    if ndict['batchsize'] is not None:
        outname += 'bs'+str(ndict['batchsize'])
    
    
    if ndict['kernel_lr'] != ndict['lr'] and ndict['kernel_lr'] is not None:
        outname+='-ka'+str(ndict['kernel_lr'])
    if ndict['adjust_lr'] != 'None':
        outname+='-'+str(ndict['adjust_lr'])[:3]
    

    
    if 'cnn_embedding' in ndict:
        if isinstance(ndict['cnn_embedding'], list):
            if len(np.unique(ndict['cnn_embedding']))>1:
                cnnemb = '-'.join(np.array(unique_ordered(ndict['cnn_embedding'])).astype(str))
            else:
                cnnemb = ndict['cnn_embedding'][0]
        else:
            cnnemb = ndict['cnn_embedding']
            
        if isinstance(ndict['n_combine_layers'], list):
            if len(np.unique(ndict['n_combine_layers']))>1:
                nclay = ''.join(np.array(ndict['n_combine_layers']).astype(str))
            else:
                nclay = ndict['n_combine_layers'][0]
        else:
            nclay = ndict['n_combine_layers']
            
        outname += '_comb'+str(cnnemb)+'nl'+str(nclay)+ndict['combine_function']+str(ndict['combine_widening'])+'r'+str(ndict['combine_residual'])
    
    
    
    return outname

    

    
