# compare predictions

# Look at spearman correlation of predictors for
# 1. performance for conditions
# 2. performance for genes
# 3. importance for motifs

# compare against each other: (make matrix visualization)
    #best CNN - kmer-model - motif-model : on 1. and 2.
    #k-mermodel of cds - promoter - 3utr: on 1., 2.
    
# for cds, 3utr and promoter
    #individually compare correlation between
        #cell types: on 3. 
        #cytokines on 3
    
    # Look at
        #cell types: mean performance
        #cytokines: mean performance


import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from functools import reduce
from scipy.stats import pearsonr

def isint(pint):
    try:
        return int(pint)
    except:
        return None

def read_in(file, delimiter = ' ', splitaxis = None, splitname = '.', splitposition = None, nan = 1.):
    xaxis = []
    yaxis = []
    data = []
    obj = open(file, 'r')
    for l, line in enumerate(obj):
        if '#' in line:
            xaxis = line.strip('#').strip().split(delimiter)
        else:
            line = line.strip().split(delimiter)
            yaxis.append(line[0])
            data.append(line[1:])
    data = np.array(data, dtype = float)
    data = data.T
    xaxis = np.array(xaxis)
    yaxis = np.array(yaxis)
    data[np.isnan(data)] = nan
    if splitaxis is not None:
        groupnames = []
        if splitaxis == 0:
            data = data.T
            for name in yaxis:
                groupnames.append(name.split(splitname)[splitposition])
        elif splitaxis == 1:
            for name in xaxis:
                groupnames.append(name.split(splitname)[splitposition])            
        groupnames = np.array(groupnames)
        groups = np.unique(groupnames)
        ndata = []
        for group in groups:
            ndata.append(data[groupnames == group])
        if splitaxis == 0:
            xaxis = np.copy(yaxis)
            yaxis = groups
        else:
            yaxis = groups
        data = ndata
    
    return xaxis, yaxis, data

from scipy.cluster import hierarchy
import scipy.special as special 

def group_matplot(groups, data, correlation = 'spearman', transform = False, text = True, vmin = 0.):    
    matrix = np.zeros((len(groups),len(groups)))
    for g, gdat in enumerate(groups):
        for f, fdat in enumerate(groups[g:]):
            d1, d2 = data[g], data[f+g]
            if correlation == 'spearman':
                d1, d2 = np.argsort(np.argsort(d1,axis = 1),axis = 1), np.argsort(np.argsort(d2,axis = 1),axis = 1)
            corr = 1. - cdist(d1,d2, 'correlation')   
            if transform:
                ### USE p-values instead
                # Correlation coeffiecient follows beta distribution
                ab = len(d1[0])/2. - 1
                corr = 2*special.btdtr(ab, ab, 0.5*(1 - np.absolute(corr)))
                corr = -np.log10(corr)
            if f == 0:
                meancorr = np.mean(corr[np.triu_indices(len(corr),1)])
            else:
                meancorr = np.mean(corr[~np.isinf(corr)])
            matrix[g, f+g] = matrix[f+g,g] = meancorr

    fig = plt.figure(figsize = (0.7*len(groups),0.7*len(groups)), dpi = 150)
    ax = fig.add_subplot(121)
    ax.set_position([0.1,0.1,0.7,0.7])
    if len(groups) > 2:
        ax2 = fig.add_subplot(122)
        ax2.set_position([0.1,0.81,0.7,0.1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        #ax2.spines['left'].set_visible(False)
        dmat = 1.-matrix/np.amax(matrix)
        Z = hierarchy.linkage(dmat[np.triu_indices(len(dmat),1)], 'single')
        dn = hierarchy.dendrogram(Z, orientation = 'top', no_labels = True, above_threshold_color = 'k', ax = ax2, color_threshold = 0)
        sort = dn['leaves']
        print( sort, len(matrix))
    else:
        sort = np.arange(len(matrix),dtype = int)
    ax.imshow(matrix[sort][:,sort], cmap = 'Blues', vmin = vmin, vmax = 1)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups[sort], rotation = 90)
    ax.tick_params(left = False, labelleft = False)
    matrix = matrix[sort][:,sort]
    if text:
        for g, gi in enumerate(matrix):
            for f, fi in enumerate(gi):
                if fi < 0.1:
                    cola = 'grey'
                else:
                    cola = 'w'
                ax.text(g,f,str(np.around(fi,2)), color = cola, ha = 'center', va = 'center')
    return fig





def mean_plot(groups, data):
    lengroups = [len(g) for g in data]
    meangroups = [np.mean(g) for g in data]
    mlen = np.amax(lengroups)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sort = np.argsort(meangroups)
    if mlen > 4:
        ax.boxplot(list(data), positions = np.argsort(sort))
        for s, so in enumerate(sort):
            ax.scatter(np.ones(len(data[so]))*s+np.random.random(len(data[so]))*0.6-0.3, data[so], c='k', alpha = 0.5)
    ax.scatter(np.arange(len(meangroups)), np.array(meangroups)[sort], color = 'r', marker = 'D')
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups[sort])
    return fig


def sim_matplot(groups, features, data, group_annotation = None, group_distance = 'pearson', group_linkage = 'single', feature_distance = 'euclidean', feature_linkage = 'average', vmin = 0., vmax = 1., norm_data = True, top_bottom = 50, text = False, y_font = 7, max_vertical = 220, correlation_mat = False): 
    
    if norm_data:
        data = data/np.amax(np.absolute(data))
        print('Normed data', np.amax(data), np.amin(data))
    
    if len(features) > top_bottom and correlation_mat == False:
        rank = np.argsort(np.argsort(-data , axis = 1),axis = 1)
        rank[data == 0] = len(data[0])
        mask = np.where(np.sum(rank<top_bottom, axis = 0) > 0)[0]
        rank2 = np.argsort(np.argsort(data , axis = 1),axis = 1)
        rank2[data == 0] = len(data[0])
        mask = np.union1d(np.where(np.sum(rank2<top_bottom, axis = 0) > 0)[0], mask)
        data = data[:, mask]
        features = features[mask]
        print( 'DATA reduced to', len(features), 'most important features')
    if correlation_mat:
        figlen = len(groups)
    elif len(features) > max_vertical:
        figlen = max_vertical
        text = False
    else:
        figlen = len(features)
        print('Figlen', figlen)
    
    pwmheight = 0.7*0.8/figlen
    pwmwidth = 3.*0.7*0.8/figlen*figlen/len(groups)
    
    dendroheight = 6./figlen
    dendrowidth = 0.8
    
    fig = plt.figure(figsize = (0.4*len(groups), 0.4*figlen), dpi = 70)
    ax = fig.add_subplot(121)
    ax.set_position([0.1,0.15,0.8,0.7])
    
    ax2 = fig.add_subplot(122)
    ax2.set_position([0.1,0.85 + 0.05*dendroheight,dendrowidth,dendroheight])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    datgroups = np.copy(data)
    if group_distance == 'spearman' or group_distance == 'pearson':
        if group_distance == 'spearman':
            datgroups = np.argsort(np.argsort(-datgroups, axis = 1), axis = 1)
        group_distance = 'correlation'
        
    group_matrix = cdist(datgroups, datgroups, group_distance)
    
    if correlation_mat:
        data = 1. - group_matrix
        features = groups
    Z_groups = hierarchy.linkage(group_matrix[np.triu_indices(len(group_matrix),1)], group_linkage)
    with plt.rc_context({'lines.linewidth': 2.}):
        dn_groups = hierarchy.dendrogram(Z_groups, orientation = 'top', no_labels = True, above_threshold_color = 'k', ax = ax2, color_threshold = 0)
    ax2.grid(color = 'grey', lw = 0.8)
    sort_groups = dn_groups['leaves']
    
    datfeats = np.copy(data)
    if correlation_mat:
        sort_feats = sort_groups
    elif feature_distance is None or feature_linkage is None:
        sort_feats = np.argsort(-np.amax(datfeats, axis = 0))
    else:
        ax3 = fig.add_subplot(811)
        ax3.set_position([0.1-pwmwidth-15/len(groups),0.15,15/len(groups),0.7])
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        if feature_distance == 'spearman' or feature_distance == 'pearson':
            if feature_distance == 'spearman':
                datfeats = np.argsort(np.argsort(-datfeats, axis = 0), axis = 0)
            feature_distance = 'correlation'
            
        feat_matrix = cdist(datfeats.T, datfeats.T, feature_distance)
        
        Z_feats = hierarchy.linkage(feat_matrix[np.triu_indices(len(feat_matrix),1)], feature_linkage)
        with plt.rc_context({'lines.linewidth': 2.}):
            dn_feats = hierarchy.dendrogram(Z_feats, orientation = 'left', no_labels = True, ax = ax3, color_threshold = 0, above_threshold_color = 'k')
        sort_feats = dn_feats['leaves']
    
    ax.imshow(data.T[sort_feats][:,sort_groups], cmap = 'RdBu', vmin = vmin, vmax = vmax, aspect = 'auto')
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups[sort_groups], rotation = 90)
    if text:
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features[sort_feats], fontsize = y_font+1)
    data = data.T[sort_feats][:,sort_groups]
    if text:
        for g, gi in enumerate(data):
            for f, fi in enumerate(gi):
                if fi < (vmin+2*(vmax-vmin)/3) and fi > (vmin+(vmax-vmin)/3):
                    cola = 'grey'
                else:
                    cola = 'w'
                if fi != 0:
                    ax.text(f,g,str(np.around(fi,2)), color = cola, ha = 'center', va = 'center', fontsize = y_font)
           
    if group_annotation is not None:
        ax.tick_params(bottom = True, labelbottom = False)
        group_colors = np.zeros(np.shape(group_annotation))
        for g, ganno in enumerate(group_annotation):
            for u, un in enumerate(np.unique(ganno)):
                group_colors[g, np.array(ganno) == un] = u
        
        ax4 = fig.add_subplot(1,8,1)
        ax4.set_position([0.1, 0.15-(len(group_annotation)*0.7+0.1)/figlen, 0.8, len(group_annotation)*0.7/figlen])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.tick_params(labelleft = False, left = False, labelbottom = True, bottom = False)
        ax4.imshow(group_colors[:,sort_groups], cmap = 'tab20', aspect = 'auto')
        ax4.set_xticks(np.arange(len(groups)))
        ax4.set_xticklabels(groups[sort_groups], rotation = 90)
        for g, ganno in enumerate(np.array(group_annotation)[:,sort_groups]):
            if g < len(group_annotation)-1:
                ax4.plot([-0.5, len(ganno)-0.5], [g+0.5, g+0.5], c = 'w', lw = 4.)
            for h, ga in enumerate(ganno):
                ax4.text(h, g, ga, color = 'k', ha= 'center', va = 'center', fontsize = 8)
        
    return fig

if __name__ == '__main__':
        
    files = sys.argv[1]

    correlation = 'spearman'
    if '--correlation' in sys.argv:
        correlation = sys.argv[sys.argv.index('--correlation')+1]
        

    if ',' in files:
        groups = np.array(sys.argv[2].split(','))
        files = files.split(',')
        data = []
        ys = []
        xs = []
        for fi in files:
            xnames, ynames, datmat = read_in(fi)
            xs.append(xnames)
            ys.append(ynames)
            data.append(datmat)
        
        cy = reduce(np.intersect1d, ys)
        ndata = []
        for d, dat in enumerate(data):
            print( np.shape(dat))
            ndata.append(np.array(dat)[:, np.argsort(ys[d])][:, np.isin(np.sort(ys[d]), cy)])
        
        data = np.array(ndata)
        
        fig = group_matplot(groups, data, correlation = correlation)
        
    else:
        splitaxis = isint(sys.argv[2])
        splitposition = int(sys.argv[3])
        xnames, groups, data = read_in(files, splitaxis = splitaxis, splitposition = splitposition)
        mask = np.sum(np.absolute(data), axis = 0) > 0
        groups = groups[mask]
        data = data[:, mask]
        mask = np.sum(np.absolute(data), axis = 1) > 0
        xnames = xnames[mask]
        data = data[mask]
        print( xnames, groups, np.shape(data))
        if splitaxis is None:
            groupanno = [[],[]]
            for gr in xnames:
                groupanno[0].append(gr.split('.')[0])
                groupanno[1].append(gr.split('.')[-1])
            sys.setrecursionlimit(10000)
            fig = sim_matplot(xnames, groups, data, group_annotation = groupanno, group_distance = 'pearson', group_linkage = 'single', feature_distance = 'euclidean', feature_linkage = 'average', vmin = -1., vmax = 1., norm_data = False, top_bottom = splitposition, text = True, correlation_mat = True)
        
        elif np.shape(data[0])[1] == 1:
            fig = mean_plot(groups, data)
        else:            
            fig = group_matplot(groups, data, correlation = correlation)
    
    if '--savefig' in sys.argv:
        figname = sys.argv[sys.argv.index('--savefig')+1]
        fig.savefig(figname, bbox_inches = 'tight', dpi = 150)
    else:
        plt.show()
    
    
    







    
    
