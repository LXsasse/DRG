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
            print(tclass, 'AUPRC', metrics.average_precision_score(Ytest[:,consider], Y_pred[:,consider]))
    if '--mseaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'MSE', mse(Ytest[:,consider], Y_pred[:,consider]))
    if '--correlationaverage' in sysargv:
        for tclass in np.unique(testclasses):
            consider = np.where(testclasses == tclass)[0]
            print(tclass, 'Correlation classes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 0)))
            print("Mean between classes", np.mean(cdist(Ytest[:,consider].T, Ytest[:,consider].T, 'correlation')[np.triu_indices(len(consider) ,1)]))
            print("Correlation to Mean", np.mean(correlation(Ytest[:,consider].T, np.array(len(consider)*[np.mean(Ytest[:,consider],axis =1)]), axis = 0)))
            print(tclass, 'Correlation genes', np.mean(correlation(Ytest[:,consider], Y_pred[:,consider], axis = 1)))
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
            Y_predc[Y_predc < 0] = 0
            JShannon = np.mean(np.transpose(jensenshannon(np.transpose(Ytest[:,consider]), np.transpose(Y_predc[:,consider]))), axis = 0)
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
            Y_predc[Y_predc < 0] = 0
            JShannon = np.mean(np.transpose(jensenshannon(np.transpose(Ytest[:,consider]), np.transpose(Y_predc[:,consider]))), axis = 1)
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
       

        
def add_params_to_outname(outname, ndict):
    
    outname += '_lf'+ndict['loss_function'][:3]+ndict['loss_function'][max(3,len(ndict['loss_function'])-3):]+'nk'+str(ndict['num_kernels'])+'l'+str(ndict['l_kernels'])+str(ndict['kernel_bias'])[0]+'kf'+ndict['kernel_function']
    
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
       
    if 'reverse_complement' in ndict:
        if ndict['reverse_complement']:
            outname += 'rcT'
    
    if 'l_out' in ndict:
        if ndict['l_out'] is None:
            outname += 'bp1'
        outname += 'bp'+str(int(ndict['l_seqs']/ndict['l_out']))
    
    if ndict['validation_loss'] != ndict['loss_function'] and ndict['validation_loss'] is not None:
        outname +='vl'+str(ndict['validation_loss'])[:2]+str(ndict['validation_loss'])[max(2,len(str(ndict['validation_loss']))-2):]
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
                
                
        if 'finalmaxpooling_size' in ndict:
            if ndict['finalmaxpooling_size'] > 0:
                outname += 'fmap'+str(ndict['finalmaxpooling_size'])
        if 'finalmeanpooling_size' in ndict:
            if ndict['finalmeanpooling_size'] > 0:
                outname += 'fmep'+str(ndict['finalmeanpooling_size'])
        
        
    if ndict['dilated_convolutions'] > 0:
        outname += '_dc'+str(ndict['dilated_convolutions'])+'i'+str(ndict['conv_increase']).strip('0').strip('.')+'d'+str(ndict['dilations']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(ndict['strides']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','') +'l'+str(ndict['l_dilkernels'])
        if ndict['dilmax_pooling']:
            outname += 'p'+str(ndict['dilmax_pooling'])[0]+str(ndict['dilmean_pooling'])[0]
        if ndict['dilpooling_size'] != ndict['pooling_size'] and ndict['dilpooling_size'] is not None:
            outname += str(ndict['dilpooling_size'])
        if ndict['dilpooling_steps'] != ndict['pooling_steps'] and ndict['dilpooling_steps'] is not None:
            outname += str(ndict['dilpooling_steps'])
        if ndict['dilpooling_residual'] > 0:
            outname += 'r'+str(ndict['dilpooling_residual'])
        if ndict['dilresidual_entire'] > 0:
            outname += 're'
    
    if ndict['embedding_convs'] > 0:
        outname += 'ec'+str(ndict['embedding_convs'])
    
    if ndict['n_transformer'] >0:
        outname += 'trf'+str(ndict['sum_attention'])[0]+str(ndict['n_transformer'])+'h'+str(ndict['n_distattention'])+'-'+str(ndict['dim_distattention'])
    
    elif ndict['n_attention'] > 0:
        outname += 'at'+str(ndict['n_attention'])+'h'+str(ndict['n_distattention'])+'-'+str(ndict['dim_distattention'])+'m'+str(ndict['maxpool_attention'])
        if ndict['dim_embattention'] is not None:
            outname += 'v'+str(ndict['dim_embattention'])
    
    if ndict['transformer_convolutions'] > 0:
        outname += '_tc'+str(ndict['transformer_convolutions'])+'d'+str(ndict['trconv_dim'])+'d'+str(ndict['trdilations']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')+'s'+str(ndict['trstrides']).replace(' ', '').replace(',', '-').replace('[','').replace(']','').replace('(','').replace(')','')
        
        if ndict['trpooling_residual'] > 0:
            outname += 'r'+str(ndict['trpooling_residual'])
        if ndict['trresidual_entire'] > 0:
            outname += 're'
      
        outname+='l'+str(ndict['l_trkernels'])
        if ndict['trmax_pooling']:
            outname += 'p'+str(ndict['trmax_pooling'])[0]+str(ndict['trmean_pooling'])[0]
        if ndict['trpooling_size'] != ndict['pooling_size'] and ndict['trpooling_size'] is not None:
            outname += str(ndict['dilpooling_size'])
        if ndict['trpooling_steps'] != ndict['pooling_steps'] and ndict['trpooling_steps'] is not None:
            outname += str(ndict['trpooling_steps'])
    
    if 'nfc_layers' in ndict:
        if ndict['nfc_layers'] > 0:
            outname +='_nfc'+str(ndict['nfc_layers'])+ndict['fc_function']+'r'+str(ndict['nfc_residuals'])
    if 'interaction_layer' in ndict:
        if ndict['interaction_layer']:
            outname += '_intl'+str(ndict['interaction_layer'])[0]
    elif 'neuralnetout' in ndict:
        if ndict['neuralnetout'] > 0:
            outname += 'nno'+str(ndict['neuralnetout'])
    if 'final_kernel' in ndict:
        outname += str(ndict['final_kernel'])+'-'+str(ndict['final_strides'])+str(ndict['predict_from_dist'])[0]
    
    
    if 'outclass' in ndict:
        if ndict['outclass'] != 'Linear':
            outname += ndict['outclass'][:4]+ndict['outclass'][-min(4,len(ndict['outclass'])-4):]
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
        outname += '_comb'+str(ndict['cnn_embedding'])+'nl'+str(ndict['n_combine_layers'])+ndict['combine_function']+str(ndict['combine_widening'])+'r'+str(ndict['combine_residual'])
    
    
    
    return outname

    

    
