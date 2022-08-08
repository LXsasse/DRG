import numpy as np
import sys, os
import glob
import matplotlib.pyplot as plt
import ast
import scipy.stats as st
from functools import reduce 


def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool

def check(inbool):
    if inbool == 'True' or inbool == 'TRUE' or inbool == 'true':
        return 1
    elif inbool == 'False' or inbool == 'FALSE' or inbool == 'false':
        return 0
    elif inbool == 'None' or inbool == 'NONE' or inbool == 'none':
        return 'None'
    #elif "[" in inbool or "(" in inbool:
        #return ast.literal_eval(inbool)
    else:
        inbool = numbertype(inbool)
    return inbool

important_params = np.array(['loss_function', 'validation_loss', 'num_kernels', 'kernel_bias', 'fixed_kernels', 'motif_cutoff', 'l_kernels', 'kernel_function', 'hot_start', 'kernel_thresholding', 'max_pooling', 'mean_pooling', 'pooling_size', 'pooling_steps', 'dilated_convolutions', 'strides', 'conv_increase', 'dilations', 'l_dilkernels', 'dilmax_pooling', 'dilmean_pooling', 'dilpooling_size', 'dilpooling_steps', 'dilpooling_residual', 'dilresidual_entire', 'gapped_convs', 'gapconv_residual', 'gapconv_pooling', 'embedding_convs', 'n_transformer', 'n_attention', 'n_distattention', 'dim_distattention', 'dim_embattention', 'maxpool_attention', 'sum_attention', 'transformer_convolutions', 'trdilations', 'trstrides', 'l_trkernels', 'trconv_dim', 'trmax_pooling', 'trmean_pooling', 'trpooling_size', 'trpooling_steps', 'trpooling_residual', 'nfc_layers', 'nfc_residuals', 'fc_function', 'layer_widening', 'interaction_layer', 'neuralnetout', 'dropout', 'batch_norm', 'l1_kernel', 'l2reg_last', 'l1reg_last', 'reverse_sign', 'shift_sequence', 'random_shift', 'smooth_onehot', 'lr', 'kernel_lr', 'adjust_lr', 'batchsize', 'outclass', 'optimizer', 'optim_params', 'seed', 'restart', 'cnn_embedding', 'n_inputs', 'n_combine_layers', 'combine_function', 'combine_widening', 'combine_residual'])


param_files = glob.glob(sys.argv[1]+'*'+sys.argv[2])
corrfile = sys.argv[3]

osoutname = False
if '--osoutname' in sys.argv:
    osoutname = True

filenames = []
all_params = []
parameters = []
performance = []
permeasures = []
for p, pfile in enumerate(param_files):
    obj = open(pfile, 'r').readlines()
    params = {}
    dnot = True
    falselines = []
    outname = ''
    for l, line in enumerate(obj):
        line = line.strip().split(' : ')
        if line[0] != 'outname':
            if line[0] in important_params:
                if len(line) > 0:
                    params[line[0]] = check(line[1])
                    all_params.append(line[0])
                else:
                    dnot=False
                    falselines.append(line)
                    break
        else:
            if osoutname:
                outname = pfile.replace(sys.argv[2], '')
            else:
                outname = os.path.split(line[1])[1]
    if os.path.isfile(outname + corrfile) and dnot:
            
        parameters.append(params)
        
        perform = []
        permeasure = []
        pobj = open(outname + corrfile, 'r').readlines()
        for l, line in enumerate(pobj):
            if line[0] != '#':
                perform.append(float(line.strip().split()[1]))
                permeasure.append(line.strip().split()[0])
        performance.append(perform)
        permeasures.append(permeasure)
        filenames.append(outname)
    else:
        if not dnot:
            print( pfile, falselines)
        elif not os.path.isfile(outname + corrfile):
            print( '\nfor', pfile)
            print( outname + corrfile, 'not a file')


uniper = reduce(np.intersect1d, permeasures)
print( 'Sorted performances', np.shape(performance), len(uniper))
if '--unmatched_data' in sys.argv:
    performance = np.array(performance)
else:
    nperformance = []
    for p, permeasure in enumerate(permeasures):
        nperformance.append(np.array(performance[p])[np.argsort(permeasure)][np.isin(np.sort(permeasure), uniper)])
    performance = np.array(nperformance)
    performance[np.isnan(performance)] = 1.

print( np.shape(performance))

print(len(parameters))

all_params = np.unique(all_params)
mean_performance = [np.mean(perform) for perform in performance]
sort_performance = np.argsort(mean_performance)
rank_performance = np.argsort(sort_performance)


print( '\nBest performing models')
for si, s in enumerate(sort_performance):
    print( si +1, mean_performance[s], filenames[s])



keep = []
for a, apar in enumerate(important_params):
    yval = []
    for i in range(len(parameters)):
        if apar in parameters[i].keys():
            yval.append(parameters[i][apar])
        else:
            yval.append('None')
            parameters[i][apar] = 'None'
    keep.append(len(np.unique(yval)))
    print(apar, keep[-1])

def transform(pval):
    if pval<0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    return ''


keep = np.array(keep, dtype = float)
important_params = important_params[keep>1]
keep = keep[keep>1]

persize = 12.
rows = len(important_params)+2
fig = plt.figure(figsize = (len(sort_performance)*0.3, (np.sum(keep)+persize)*0.3), dpi = 60)
axpf = fig.add_subplot(rows, 1, 1)
axpf.set_position([0.05, 0.95-(persize/(np.sum(keep)+persize))*0.9, 0.9, (persize/(np.sum(keep)+persize))*0.9])
axpf.scatter(rank_performance, mean_performance, color = 'r')
axpf.plot([-0.5, len(mean_performance)-0.5], [np.amin(mean_performance), np.amin(mean_performance)], color = 'orange', ls = '--')
if len(performance[0])>4:
    axpf.boxplot(list(performance), positions = rank_performance)
#axpf.set_ylim([np.amin(np.concatenate(performance)), np.amax(mean_performance)])
axpf.spines['right'].set_visible( False)
axpf.spines['top'].set_visible(False)
axpf.tick_params(labeltop = True, labelbottom = False, bottom = False, top = False)
axpf.set_xticks(np.arange(len(mean_performance), dtype = int))
pvalticks = ['<']
for i, p in enumerate(sort_performance[1:]):
    if '--unmatched_data' in sys.argv:
        pval = st.mannwhitneyu(performance[sort_performance[0]], performance[p], alternative = 'less')[1]
        pval2 = st.mannwhitneyu(performance[sort_performance[i]], performance[p], alternative = 'less')[1]
    else:
        pval = st.wilcoxon(performance[sort_performance[0]], performance[p], alternative = 'less')[1]
        if not (performance[sort_performance[i]] == performance[p]).all():
            pval2 = st.wilcoxon(performance[sort_performance[i]], performance[p], alternative = 'less')[1]
        else:
            pval2 = 1.
    pvalticks.append(transform(pval2) + '\n'+transform(pval))
axpf.set_xticklabels(pvalticks)

axpf.set_xlim([-0.5, len(mean_performance)-0.5])
axpf.grid(axis = 'both')
axpf.set_ylabel('Performance')
yticks = np.array(axpf.get_yticks())
ytickdist =np.absolute(yticks - np.amin(mean_performance))
yticks[np.argmin(ytickdist)] = round(np.amin(mean_performance),2)
axpf.set_yticks(yticks)


taken = 0
for a, apar in enumerate(important_params):
    yval = [parameters[i][apar] for i in range(len(parameters))]
    print( apar, np.unique(yval))
    ax = fig.add_subplot(rows, 1, a+2)
    ax.set_position([0.05, 0.95-((np.sum(keep[:a+1])+persize)/(np.sum(keep)+persize))*0.9, 0.9, (keep[a]/(np.sum(keep)+persize))*0.9 - 0.3/(np.sum(keep)+persize)])
    ax.spines['right'].set_visible( False)    
    ax.spines['top'].set_visible(False)
    x, y = [],[]
    for r in range(len(rank_performance)):
        if yval[r] != 'None':
            x.append(rank_performance[r])
            y.append(yval[r])
    checkarry = np.array([isinstance(yi, str) for yi in y]).any()
    if checkarry:
        y = np.array(y, dtype = str)
        unstr, unstrn = np.unique(y, return_counts = True)
        y = [list(unstr).index(yi) for yi in y]
    else:
        unstr, unstrn = np.unique(y, return_counts = True)
        y = [list(unstr).index(yi) for yi in y]
    
    ax.bar(np.sort(x), np.array(y)[np.argsort(x)]+0.5, color = 'slategrey', alpha = 0.4, width = 1., bottom = -0.5)
    ax.scatter(np.sort(x), np.array(y)[np.argsort(x)], marker = 'D', s = 30, color = 'darkslategrey')
    #ax.step(np.sort(x), np.array(y)[np.argsort(x)], where ='mid', linestyle = '--')
    ax.set_yticks(np.unique(y))
    ax.set_yticklabels(unstr)
    ax.set_xticks(np.arange(len(mean_performance), dtype = int))
    ax.set_xlim([-0.5, len(mean_performance)-0.5])
    ax.set_ylim([-0.5, np.amax(y)+0.5])
    ax.grid(axis = 'both')
    ax.text(ax.get_xlim()[1], (ax.get_ylim()[1] + ax.get_ylim()[0])/2., apar, ha = 'left', va = 'center')
    if a == len(important_params) -1:
        ax.set_xticklabels(np.arange(len(mean_performance), dtype = int), rotation = 90)
        ax.set_xlabel('Model')
    else:
        ax.tick_params(labelbottom = False)
        
    
if '--savefig' in sys.argv:
    print( 'saved as', sys.argv[1]+sys.argv[2].rsplit('.',1)[0]+sys.argv[3].rsplit('.',1)[0]+'.jpg')
    fig.savefig(sys.argv[1]+sys.argv[2].rsplit('.',1)[0]+sys.argv[3].rsplit('.',1)[0]+'.jpg', dpi = 175, bbox_inches = 'tight')
else:
    plt.show()


# Where do strides and dilations go? 
# What happens to gapped convs?




























