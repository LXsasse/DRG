# model_output.py
'''
Handles output for training and testing models.
TODO 
- clean up functions
- improve naming from dictionary
'''
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist, jensenshannon

from .stats_functions import correlation, mse, RankSum, AuROC, AuPR
from .sequence_utils import avgpool



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






            
def shuffle_along_axis(array, axis = 0):
    '''
    Shuffle elements in a multi-D array along the given axis
    '''
    if axis != 0:
        # if axis is negative make it positive, otherwise keep
        axis = int(axis < 0)*len(np.shape(array))+axis
        # number of all axes in array
        axes = [i for i in range(len(np.shape(array)))]
        del axes[axis]
        # rearrange axis so that given axis is first
        axes = [axis] + axes
        array = np.transpose(array, axes = axes)
    rng = np.random.default_rng()
    shuf = np.arange(len(array))
    rng.shuffle(shuf)
    array = array[shuf]
    if axis != 0:
        axes = np.argsort(axes)
        array = np.transpose(array, axes = axes)
    return array

    
def compute_performance(Ytest, Y_pred, func, axis, testclasses, experiments, meanclasses, compare_random = True):
    if len(np.shape(Ytest))>2:
        Ytest = np.sum(Ytest, axis = -1)
    if len(np.shape(Y_pred))>2:
        Y_pred = np.sum(Y_pred, axis = -1)
    out = {}
    for tclass in np.unique(testclasses):
        pvalue = None
        consider = np.where(testclasses == tclass)[0]
        exp = experiments[consider]
        yt = Ytest[:,consider]
        yp = Y_pred[:,consider]
        if meanclasses is not None:
            exp = np.unique(meanclasses[consider])
            yt = np.array([np.mean(yt[:, meanclasses[consider] == mclass], axis = 1) for mclass in np.unique(meanclasses[consider])]).T
            yp = np.array([np.mean(yp[:, meanclasses[consider] == mclass], axis = 1) for mclass in np.unique(meanclasses[consider])]).T
        metric = func(yt, yp, axis = axis)
        if compare_random and len(metric) > 10:
            axis = int(axis < 0)*len(np.shape(yt))+axis
            random = func(shuffle_along_axis(yt, axis = 1-axis), yp, axis = axis)
            pvalue = stats.wilcoxon(metric, random, alternative = 'less')[1] # determines p-value that random is less than metric although H0 rejected
        out[tclass] = (metric, exp, pvalue)
    return out

def outfmt(names, experiments, values):
    if len(names) == len(values):
        return np.append(names.reshape(-1,1), values.reshape(-1,1), axis = 1)
    elif len(experiments) == len(values):
        return np.append(experiments.reshape(-1,1), values.reshape(-1,1), axis = 1)

    
def save_performance(Y_pred, Ytest, testclasses, experiments, names, outname, sysargv, compare_random = True, meanclasses = None, pooling = None):
    evaldict = {'--save_mse_perclass':(mse, '_clss_mse_tcl', 0, None), 
                '--save_poolmse_perclass':(mse, '_clss_mse_tcl', 0, 10), 
                '--save_jensenshannon_perclass':(jensenshannon, '_clss_js_tcl', None), 
                '--save_wilcoxon_perclass': (RankSum,'_clss_rnksm_tcl', 0, None), 
                '--save_correlation_perclass':(correlation, '_clss_corr_tcl', 0, None), 
                '--save_auroc_perclass':(AuROC, '_clss_auroc_tcl',0, None), 
                '--save_auprc_perclass':(AuPR, '_clss_auprc_tcl',0, None), 
                '--save_mse_perpoint':(mse, '_pnt_mse_tcl', 1, None), 
                '--save_poolmse_perpoint':(mse, '_pnt_mse_tcl', 1, 10), 
                '--save_jensenshannon_perpoint':(jensenshannon, '_pnt_js_tcl',1, None), 
                '--save_wilcoxon_perpoint': (RankSum,'_pnt_rnksm_tcl', 1, None), 
                '--save_correlation_perpoint':(correlation, '_pnt_corr_tcl', 1, None), 
                '--save_auroc_perpoint':(AuROC, '_pnt_auroc_tcl',1, None), 
                '--save_auprc_perpoint':(AuPR, '_pnt_auprc_tcl', 1, None),
                }
    for sarg in sysargv:
        if sarg in evaldict:
            func, nameadd, axis, pooling = evaldict[sarg]
            print(sarg)
            if pooling is not None:
                Ytest = avgpool(Ytest, pooling)
                Y_pred = avgpool(Y_pred, pooling)
                nameadd = '_pool'+str(pooling)+nameadd
            
            result = compute_performance(Ytest, Y_pred, func, axis, testclasses, experiments, meanclasses, compare_random = compare_random)
            for r in result:
                res = result[r] 
                np.savetxt(outname+nameadd+r+'.txt', outfmt(names, res[1], res[0]), fmt = '%s')


 
 
def unique_ordered(alist):
    a = []
    for l in alist:
        if l not in a:
            a.append(l)
    return a




def add_params_to_outname(outname, ndict):
    '''
    Generates outname for neural networks from model.__dict__
    '''
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
    if 'fc_function' in ndict:
        if ndict['fc_function'] != ndict['net_function'] and nfcgiven:
            outname += ndict['fc_function']
    if 'fclayer_size' in ndict:
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
    
    if 'conv_dropout' in ndict:
        if ndict['conv_dropout'] > 0.:
            outname += 'cdo'+str(ndict['conv_dropout'])
    if 'conv_batch_norm' in ndict:
        if ndict['conv_batch_norm']:
            outname += 'cbno'+str(ndict['conv_batch_norm'])[0]
    if 'attention_dropout' in ndict:
        if ndict['attention_dropout'] > 0.:
            outname += 'ado'+str(ndict['attention_dropout'])
    if 'attention_batch_norm' in ndict:
        if ndict['attention_batch_norm']:
            outname += 'abno'+str(ndict['attention_batch_norm'])[0]
    if 'fc_dropout' in ndict:
        if ndict['fc_dropout'] > 0.:
            outname += 'fdo'+str(ndict['fc_dropout'])
    if 'fc_batch_norm' in ndict:
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
        cnnemb = ''
        if 'shared_embedding' in ndict:
            if ndict['shared_embedding']:
                cnnemb += 'se'
        
        if isinstance(ndict['cnn_embedding'], list):
            if len(np.unique(ndict['cnn_embedding']))>1:
                cnnemb += '-'.join(np.array(unique_ordered(ndict['cnn_embedding'])).astype(str))
            else:
                cnnemb += str(ndict['cnn_embedding'][0])
        else:
            cnnemb += str(ndict['cnn_embedding'])
       
        if isinstance(ndict['n_combine_layers'], list):
            if len(np.unique(ndict['n_combine_layers']))>1:
                nclay = ''.join(np.array(ndict['n_combine_layers']).astype(str))
            else:
                nclay = ndict['n_combine_layers'][0]
        else:
            nclay = ndict['n_combine_layers']
        
        if isinstance(ndict['combine_function'], list):
            if len(np.unique(ndict['combine_function']))>1:
                cof = '-'.join(np.array(unique_ordered(ndict['combine_function'])).astype(str))
            else:
                cof = str(ndict['combine_function'][0])
        else:
            cof = str(ndict['combine_function'])
            
        outname += '_comb'+str(cnnemb)+'nl'+str(nclay)+cof+str(ndict['combine_widening'])+'r'+str(ndict['combine_residual'])
    
    
    
    return outname

    

    
