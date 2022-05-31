import os
import sys
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logomaker
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as hierarchy


# Check if convolutions are similar for:
    #1. Seeds
    #2. Different folds
    #3. Different CNN architectures


def generate_pwms_from_seqs(seqs, activation = None, act_cut = None, weighted = False):
    if weighted is not None and activation is not None:
        seqs = seqs * activation[:, None, None]
    
    if activation is not None and act_cut is not None:
        seqs = seqs[activation > act_cut]
    
    ppm = np.sum(seqs, axis = 0)
    ppm = ppm/np.sum(ppm, axis = 1)
    
    return ppm

def compare_ppms(ppms, ppms_ref, find_bestmatch = True, fill_logp_self = 0, one_half = True, min_sim = 5, padding = 0.25, infocont = False, bk_freq = 0.25):
    # measure lengths of motifs
    motif_len, motif_len_ref = [np.shape(ppm)[0] for ppm in ppms], [np.shape(ppm)[0] for ppm in ppms_ref]
    offsets = np.zeros((len(ppms), len(ppms_ref)), dtype = int)
    log_pvalues = np.zeros((len(ppms), len(ppms_ref)), dtype = float)
    correlation = np.zeros((len(ppms), len(ppms_ref)), dtype = float)
    if infocont:
        padding = max(0,np.log2(padding/bk_freq))
    for p, ppm in enumerate(ppms):
        if p% 25 == 0:
            print(p)
        if one_half: 
            p_start = p+1
        else:
            p_start = 0
        if infocont:
            ppm = np.log2(ppm/bk_freq)
            ppm[ppm<0] = 0
        for q in range(p_start, len(ppms_ref)):
            qpm = ppms_ref[q]
            if infocont:
                qpm = np.log2(qpm/bk_freq)
                qpm[qpm<0] = 0
            pvals = []
            pearcors = []
            offs = []
            
            #vectors = []
            #shorts = []
            for i in range(min(0,-motif_len[p]+min_sim), motif_len_ref[q]-min(motif_len[p],min_sim) + 1):
                if padding is not None:
                    refppm, testppm = np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding, np.ones((motif_len_ref[q]-min(motif_len[p],min_sim) + motif_len[p]-min(0,-motif_len[p]+min_sim),4))*padding
                    refppm[-min(0,-motif_len[p]+min_sim):motif_len_ref[q]-min(0,-motif_len[p]+min_sim)] = qpm
                    testppm[i-min(0,-motif_len[p]+min_sim):i-min(0,-motif_len[p]+min_sim)+motif_len[p]] = ppm
                    mask = np.sum(testppm+refppm != 2*padding ,axis = 1) > 0
                    refppm, testppm = refppm[mask], testppm[mask]
                else:
                    refppm = qpm[max(i,0):min(motif_len_ref[q],motif_len[p]+i)]
                    testppm = ppm[max(0,-i): min(motif_len[p],motif_len_ref[q]-i)]
                peacor, pval = pearsonr(testppm.flatten(), refppm.flatten())
                #vectors.append([testppm.flatten(), refppm.flatten()])
                #shorts.append(pfm2iupac([testppm, refppm], bk_freq = 0.28))
                pvals.append(-np.sign(peacor)*np.log10(pval))
                pearcors.append(peacor)
                offs.append(i)
            maxp = np.argmax(pvals)
            log_pvalues[p,q] = pvals[maxp]
            offsets[p,q] = offs[maxp]
            correlation[p,q] = 1.-pearcors[maxp]
            if one_half:
                log_pvalues[q,p] = pvals[maxp]
                offsets[q,p] = -offs[maxp]
                correlation[q,p] = 1.-pearcors[maxp]
            #if pearcors[maxp] > 0.8:
                #print(shorts[maxp])
                #plt.scatter(vectors[maxp][0], vectors[maxp][1])
                #plt.show()
                #plt.close()
    log_pvalues[np.isinf(log_pvalues)] = fill_logp_self
    log_pvalues = log_pvalues * np.sign(correlation)
    if one_half:
        np.fill_diagonal(log_pvalues, fill_logp_self)
    
    
    if find_bestmatch or len(ppms)!=len(ppms_ref):
        bestmatch = np.argmax(log_pvalues, axis = 1)
    else:
        bestmatch = -np.ones(len(ppms), dtype = int)
        n_refs = np.arange(len(ppms_ref), dtype = int)
        n_ppms = np.arange(len(ppms_ref), dtype = int)
        asar = np.copy(log_pvalues)
        while True:
            maxr = (int(np.argmax(asar)/len(asar)), np.argmax(asar)%len(asar))
            bestmatch[n_ppms[maxr[0]]] = n_refs[maxr[1]]
            asar = np.delete(asar, maxr[0], axis = 0)
            asar = np.delete(asar, maxr[1], axis = 1)
            n_refs = np.delete(n_refs, maxr[1])
            n_ppms = np.delete(n_ppms, maxr[0])
            if len(n_refs) == 0:
                break
    return correlation, log_pvalues, offsets, bestmatch


def compute_importance(model, in_test, out_test, activation_measure = 'euclidean', pwm_in = None, normalize = True):
    n_kernels = model.num_kernels
    full_predict = model.predict(in_test, pwm_out = pwm_in)
    #activation_measures: euclidean, correlation
    full_predict = np.diagonal(cdist(full_predict.T, out_test.T, activation_measure))
    importance = []
    for n in range(n_kernels):
        mnpredict = model.predict(in_test, mask = n, pwm_out = pwm_in)
        reduce_predict = np.diagonal(cdist(mnpredict.T, out_test.T, activation_measure))
        importance.append(reduce_predict - full_predict)
    if pwm_in is not None:
        for n in range(n_kernels, n_kernels + np.shape(pwm_in)[-2]):
            mnpredict = model.predict(in_test, mask = n, pwm_out = pwm_in)
            importance.append(np.diagonal(cdist(mnpredict.T, out_test.T, activation_measure)) - full_predict)
    importance = np.array(importance)
    if normalize:
        importance = np.around(importance/np.amax(importance),4)
    return importance

def kernel_to_ppm(kernels, kernel_bias = None, bk_freq = None):
    n_kernel, n_input, l_kernel = np.shape(kernels)
    if kernel_bias is not None:
        kernels += kernel_bias[:,None,None]
    if bk_freq is None:
        bk_freq = np.ones(n_input)*np.log2(1./float(n_input))
    elif isinstance(bk_freq, float) or isinstance(bk_freq, int):
        bk_freq = np.ones(n_input)*np.log2(1./float(bk_freq))
    kernels -= bk_freq[None,:,None]
    ppms = 2.**kernels
    ppms = ppms/np.sum(ppms, axis = 1)[:,None, :]
    return ppms

def pfm2iupac(pwms, bk_freq = None):
    hash = {'A':16, 'C':8, 'G':4, 'T':2}
    dictionary = {'A':16, 'C':8, 'G':4, 'T':2, 'R':20, 'Y':10, 'S':12, 'W':18, 'K':6, 'M':24, 'B':14, 'D':22, 'H':26, 'V':28, 'N':0}
    res = dict((v,k) for k,v in dictionary.items())
    n_nts = len(pwms[0].T)
    if bk_freq is None:
        bk_freq = (1./float(n_nts))*np.ones(n_nts)
    else:
        bk_freq = bk_freq*np.ones(n_nts)
    motifs = []
    for pwm in pwms:
        m = ''
        for p in pwm:
            score = 0
            for i in range(len(p)):
                if p[i] > bk_freq[i]:
                    score += list(hash.values())[i]
            m += res[score]
        motifs.append(m)
    return motifs
        
def write_meme_file(pwm, pwmname, output_file_path, transpose = True):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    if transpose:
        pwms = []
        for pw in pwm:
            pwms.append(np.array(pw).T)
        pwm = pwms
        
    n_filters = len(pwm)
    print(n_filters)
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= ACGT \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        if np.sum(pwm[i]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n"
                % np.count_nonzero(np.sum(pwm[i], axis=0))
            )
        
        for j in range(0, np.shape(pwm[i])[-1]):
            if np.sum(pwm[i][:, j]) > 0:
                meme_file.write(
                    str(pwm[i][ 0, j])
                    + "\t"
                    + str(pwm[i][1, j])
                    + "\t"
                    + str(pwm[i][2, j])
                    + "\t"
                    + str(pwm[i][3, j])
                    + "\n"
                )

    meme_file.close()

def read_meme(file_path):
    lines = open(file_path, 'r').readlines()
    pwms = []
    names = []
    read = False
    for l, line in enumerate(lines):
        line = line.strip().split()
        if read:
            if len(line) == 0:
                pwms.append(np.array(pwm))
                read = False
            elif line[0] != 'letter-probability':
                pwm.append(np.array(line,dtype = float))
                
        if len(line) > 0:
            if line[0] == 'MOTIF':
                names.append(line[1])
                read = True
                pwm = []
        
        if l == len(lines)-1 and read:
            pwms.append(np.array(pwm))
    return pwms, names
    

def combine_pwms(pwms, clusters, similarity, offsets, maxnorm = True):
    lenpwms = np.array([len(pwm) for pwm in pwms])
    unclusters = np.unique(clusters)
    comb_pwms = []
    
    #pwmotif = np.array(pfm2iupac(pwms, bk_freq = 0.28))
    
    for u in unclusters:
        mask = np.where(clusters == u)[0]
        if len(mask) > 1:
            #print(pwmotif[mask])
            simcluster = np.argmax(np.sum(similarity[mask][:,mask], axis = 1))
            offsetcluster = offsets[mask][:,mask]
            off = offsetcluster.T[simcluster]
            #print(simcluster, off)
            clusterlen = lenpwms[mask]
            #print(similarity[mask][:, mask])
            #print(correlation[mask][:,mask])
            seed = np.zeros((np.amax(off+clusterlen)-np.amin(off),len(pwms[0][0])))
            #seed2 = np.copy(seed)
            seed[-np.amin(off):lenpwms[mask[simcluster]]-np.amin(off)] = pwms[mask[simcluster]]
            #seed1 = np.copy(seed)
            #print(pfm2iupac([seed], bk_freq = 0.28)[0])
            for m, ma in enumerate(mask):
                if m != simcluster:
                    seed[-np.amin(off)+off[m]:lenpwms[ma]+off[m]-np.amin(off)] += pwms[ma]
                    #check = np.copy(seed2)
                    #check[-np.amin(off)+off[m]:lenpwms[ma]+off[m]-np.amin(off)] = pwms[ma]
                    #print(pfm2iupac([check], bk_freq = 0.28)[0], pearsonr(check.flatten(), seed1.flatten()))
            if maxnorm:
                seed = seed/np.amax(np.sum(seed,axis = 1))
            else:
                seed = seed/np.sum(seed,axis = 1)[:, None]
            #print(pfm2iupac([seed], bk_freq = 0.28)[0])
            comb_pwms.append(seed)
        else:
            comb_pwms.append(pwms[mask[0]])
    return comb_pwms

def sim_matplot(groups, features, data, group_annotation = None, group_distance = 'spearman', group_linkage = 'single', feature_distance = None, feature_linkage = None, feature_pwms = None, text = True, vmin = 0., vmax = 1., norm_data = False, top = 100, dpi = 60, pwm_min = 0, pwm_max = 2 ): 
    
    if norm_data:
        data = data/np.amax(np.absolute(data))
    
    if len(features) > top:
        rank = np.argsort(np.argsort(-data , axis = 1),axis = 1)
        mask = np.where(np.sum(rank<top, axis = 0) > 0)[0]
        data = data[:, mask]
        features = features[mask]
        if feature_pwms is not None:
            npwms = [feature_pwms[m] for m in mask]
            feature_pwms = npwms
        print('DATA reduced to', len(features), 'most important features')
        
    pwmheight = 0.7*0.8/len(features)
    pwmwidth = 2.*0.7*0.8/len(features)*len(features)/len(groups)
    
    dendroheight = 6./len(features)
    dendrowidth = 0.8
    
    fig = plt.figure(figsize = (0.4*len(groups), 0.4*len(features)), dpi = dpi)
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
        group_distance= 'correlation'
        
    group_matrix = cdist(datgroups, datgroups, group_distance)
    
    Z_groups = hierarchy.linkage(group_matrix[np.triu_indices(len(group_matrix),1)], group_linkage)
    with plt.rc_context({'lines.linewidth': 2.}):
        dn_groups = hierarchy.dendrogram(Z_groups, orientation = 'top', no_labels = True, above_threshold_color = 'k', ax = ax2, color_threshold = 0)
    ax2.grid(color = 'grey', lw = 0.8)
    sort_groups = dn_groups['leaves']
    
    datfeats = np.copy(data)
    if feature_distance is None or feature_linkage is None:
        sort_feats = np.argsort(-np.amax(datfeats, axis = 0))
    else:
        ax3 = fig.add_subplot(811)
        ax3.set_position([0.1-pwmwidth-5/len(groups),0.15,5/len(groups),0.7])
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
            dn_feats = hierarchy.dendrogram(Z_feats, orientation = 'left', no_labels = True, above_threshold_color = 'k', ax = ax3, color_threshold = 0)
        sort_feats = dn_feats['leaves']
    
    ax.imshow(data.T[sort_feats][:,sort_groups], cmap = 'RdBu', vmin = vmin, vmax = vmax, aspect = 'auto')
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups[sort_groups], rotation = 90)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features[sort_feats])
    data = data.T[sort_feats][:,sort_groups]
    if text:
        for g, gi in enumerate(data):
            for f, fi in enumerate(gi):
                if fi < vmin + 0.3*(vmax -vmin):
                    cola = 'grey'
                else:
                    cola = 'w'
                ax.text(f,g,str(np.around(fi,2)), color = cola, ha = 'center', va = 'center', fontsize = 8)
    if feature_pwms is not None:
        ax.tick_params(left = True, labelleft = False)
        for s, si in enumerate(sort_feats):
            
            axpwm = fig.add_subplot(len(sort_feats),1,s+1)
            axpwm.set_position([0.1-pwmwidth, 0.15+0.7-0.7*(s+.9)/len(sort_feats),pwmwidth, pwmheight])
            logomaker.Logo(pd.DataFrame(np.log2(feature_pwms[si]/0.25), columns = list('ACGT')), ax = axpwm, color_scheme = 'classic')
            axpwm.set_ylim([pwm_min, pwm_max])
            axpwm.spines['top'].set_visible(False)
            axpwm.spines['right'].set_visible(False)
            axpwm.spines['left'].set_visible(False)
            #axpwm.spines['bottom'].set_visible(False)
            axpwm.tick_params(labelleft = False, labelbottom = False, bottom = False)
            
            
    if group_annotation is not None:
        ax.tick_params(bottom = True, labelbottom = False)
        group_colors = np.zeros(np.shape(group_annotation))
        for g, ganno in enumerate(group_annotation):
            for u, un in enumerate(np.unique(ganno)):
                group_colors[g, np.array(ganno) == un] = u
        
        ax4 = fig.add_subplot(1,8,1)
        ax4.set_position([0.1, 0.15-(len(group_annotation)*0.7+0.1)/len(features), 0.8, len(group_annotation)*0.7/len(features)])
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
    input_files = sys.argv[1]
    if ',' in input_files:
        input_files = input_files.split(',')
    else:
        input_files = [input_files]
    
    pwm_set, pwm_names, n_file, activities = [], [], [], []
    for i, infile in enumerate(input_files):
        print('READ', infile)
        pwms, names = read_meme(infile)
        pwm_set.append(pwms)
        print(len(pwms))
        pwm_names.append([name+'('+str(i)+')' for name in names])
        n_file.append(np.ones(len(names),dtype = int)*i)
        act_file = infile.replace('_kernel_ppms.meme', '_kernel_importance.dat')
        experiments = np.array(open(act_file, 'r').readline().strip('#').strip().split()[2:])
        activities.append(np.genfromtxt(act_file, dtype = float)[:, 2:])
    
    
    pwm_set = np.concatenate(pwm_set, axis = 0)
    pwm_names = np.concatenate(pwm_names, axis = 0)
    n_file = np.concatenate(n_file, axis = 0)
    activities = np.concatenate(activities, axis = 0)
    
    print('Total number of PWMs', len(pwm_set))
    
    if '--load_clusters' in sys.argv:
        cfile = sys.argv[sys.argv.index('--load_clusters')+1]
        outname = cfile.replace('_clusterpwms.meme', '')
        clusterpwms, pwmnames = read_meme(cfile)
        clusters = -np.ones(len(pwm_names))
        for p, pwmname in enumerate(pwmnames):
            pwmsin = np.isin(pwm_names, np.array(pwmname.split(',')))
            clusters[pwmsin] = p
        
    else:
        print('Computing similarity between pwms')
        correlation, logs, ofs, best = compare_ppms(pwm_set, pwm_set, find_bestmatch = True, fill_logp_self = 1000)
        
        linkage = sys.argv[2]
        distance_threshold = float(sys.argv[3])
        outname = sys.argv[4] + linkage+str(distance_threshold)
        
        # COMPARE RUNS:
        # cluster motifs on log pvalues from all files together
            # motifs from same run can also fall into the same cluster and therefore split importance for some sites
        print('Clustering pwms')
        #connectmat = (logs>=p_threshold).astype(int) # originally intended to use with connectivity = connectmat. Did not work as expected.
        clustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage = linkage, distance_threshold = distance_threshold).fit(correlation)
        clusters = clustering.labels_
        
        clusterpwms = combine_pwms(pwm_set, clusters, logs, ofs)
        pwmnames = []
        for c in np.unique(clusters):
            pwmnames.append(','.join(pwm_names[clusters == c]))
        write_meme_file(clusterpwms, np.array(pwmnames), outname+'_clusterpwms.meme')
    
    unique_clusters, cluster_len = np.unique(clusters, return_counts = True)
    print(len(unique_clusters), 'clusters found.\n', int(np.sum((cluster_len == 1))), 'single ones\n', np.amax(cluster_len), 'is largest')
    clustermotifs = np.array(pfm2iupac(clusterpwms, bk_freq = 0.28))
    
    print(clustermotifs[np.argsort(-cluster_len)[:10]], -np.sort(-cluster_len)[:10])
    
    # For each condition
    cluster_means = []
    for e, exp in enumerate(experiments):
        print(exp)
        # sort clusters by mean activity
        clust_activities = []
        clust_act_mean = []
        x_offset = []
        n_reprod = []
        for c in np.unique(clusters):
            in_clust = clusters == c
            clust_activities.append(activities[:,e][in_clust])
            clust_act_mean.append(np.mean(activities[:,e][in_clust]))
            x_offset.append(np.amax(n_file)/(np.amax(n_file)+1)*(n_file[in_clust]/np.amax(n_file)-0.5))
            n_reprod.append(len(np.unique(n_file[in_clust])))
        # sort means
        max_pwms = 100
        clust_act_mean = np.array(clust_act_mean)
        sort_mean = np.argsort(-clust_act_mean)[:max_pwms]
        cluster_means.append(clust_act_mean)
        n_reprod = np.array(n_reprod)
        # for each run plot all individual activity but use different colors and small difference on y-position
        # Use pylogo for consensus motif between all members of cluster
        if '--plot_importance' in sys.argv:
            fig = plt.figure(figsize = (5, len(sort_mean)*0.4), dpi = 100)
            ax = fig.add_subplot(111)
            ax.set_position([0.3,0.1,0.55, 0.8])
            ax.set_title(exp)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.scatter(clust_act_mean[sort_mean], -np.arange(len(sort_mean)), color = 'r', marker = 'D', s = 20)
            ax.set_ylim([-len(sort_mean)+0.5, 0.5])
            ax.set_xlabel('Feature importance')
            ax.plot([0,0], [0.5,-np.amax(sort_mean)-0.5] ,ls = '--', color = 'k')
            
            ax2 = fig.add_subplot(911)
            ax2.set_position([0.85,0.1,0.1, 0.8])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_ylim([-len(sort_mean)+0.5, 0.5])
            ax2.set_yticks(-np.arange(len(sort_mean)))
            ax2.tick_params(right = True, left = False, labelleft = False, labelright = True)
            ax2.set_yticklabels(cluster_len[sort_mean])
            ax2.barh(-np.arange(len(sort_mean)), n_reprod[sort_mean], color = 'grey')
            ax2.set_xticks(np.arange(1,np.amax(n_reprod)+1))
            ax2.grid(axis = 'x')
            for s, si in enumerate(sort_mean):
                ax.scatter(clust_activities[si], x_offset[si]-s, color = cm.Set2(x_offset[si]-np.amin(np.concatenate(x_offset))), s = 10)
                axpwm = fig.add_subplot(len(sort_mean),1,s+1)
                axpwm.set_position([0.1, 0.1 + 0.8*(1.-(s+.9)/len(sort_mean)), 0.2, 0.8*0.8/len(sort_mean)])
                axpwm.spines['top'].set_visible(False)
                axpwm.spines['right'].set_visible(False)
                axpwm.spines['left'].set_visible(False)
                #axpwm.spines['bottom'].set_visible(False)
                axpwm.tick_params(bottom = False, labelbottom = False, labelleft = False, left = True)
                logomaker.Logo(pd.DataFrame(np.log2(clusterpwms[si]/0.25), columns = list('ACGT')), ax = axpwm, color_scheme = 'classic')
                axpwm.set_ylim([0,0.75])
                axpwm.set_yticks([0])
            ax.grid(color = 'silver')
            fig.savefig(outname + '_kernel_importance_sets'+str(len(input_files))+'_'+exp+'.jpg', dpi = 125, bbox_inches = 'tight')
            print('SAVED AS: ', outname + '_kernel_importance_sets'+str(len(input_files))+'_'+exp+'.jpg')
            plt.close()
    
    # Save cluster means and save pwms
    cluster_means = np.array(cluster_means)
    np.savetxt(outname+'_clusteractivities.dat', np.concatenate([[pwmnames], [clustermotifs],cluster_means], axis = 0).T.astype(str), fmt = '%s', header = 'Kernel IUPAC '+' '.join(experiments))
    
    # COMPARE EXPERIMENTS:
    # create file with mean values for each cluster across conditions
        # cluster experiments with cell types and cytokine annotation on this matrix
    groupannos = [[],[]]
    for exp in experiments:
        groupannos[0].append(exp.split('.')[0])
        groupannos[1].append(exp.split('.')[-1])
    
    fig2 = sim_matplot(experiments, clustermotifs, cluster_means, group_annotation = groupannos, group_distance = 'euclidean', group_linkage = 'average', feature_distance = 'euclidean', feature_linkage = 'average', feature_pwms = clusterpwms, text = True, vmin = -1., vmax = 1, norm_data = True, top = 20, pwm_min = 0, pwm_max = 0.75)
    
    fig2.savefig(outname + '_kernel_importance_clusters'+str(len(input_files))+'.jpg', dpi = 130, bbox_inches = 'tight')
    print('SAVED AS: ', outname + '_kernel_importance_clusters'+str(len(input_files))+'.jpg')
    
    plt.close()
    

