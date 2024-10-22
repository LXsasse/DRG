
'''
plots a treee with PWMs and if given also a matrix, boxplots, or barplots with the pwms
'''

import numpy as np
import sys, os
import matplotlib.pyplot as plt 

from drg_tools.plotlib import plot_heatmap
from drg_tools.motif_analysis import torch_compute_similarity_motifs, reverse, combine_pwms
from drg_tools.io_utils import readin_motif_files, write_meme_file, inputkwargs_from_string, check, readalign_matrix_files

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering


if __name__ == '__main__':
    
    pwmfile = sys.argv[1] # File with PWMs

    # Specify output
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        dpi = 250
    else:
        outname = None
        dpi = None
    # If to save pwms and features after joining them in this script for visialization
    if '--savejoined' in sys.argv and outname is None:
        outname = os.path.splitext(pwmfile)[0]
    
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
        
    pwms,pwmnames,nts = readin_motif_files(pwmfile)
    
    # Modify pwmnames to shorten or make align to additional data
    if '--usepwmid' in sys.argv: # Use id's instead of names
        pwmnames = np.arange(len(pwmnames)).astype(str)
    elif '--clipname' in sys.argv: # replace a certain part of the name with ''
        clip = sys.argv[sys.argv.index('--clipname')+1]
        pwmnames = np.array([pwmname.replace(clip, '') for pwmname in pwmnames])
    
    if '--set' in sys.argv: # Filter set
        print('Before filter set', len(pwmnames))
        pwmset = np.genfromtxt(sys.argv[sys.argv.index('--set')+1], dtype = str)
        keep = np.where(np.isin(pwmnames, pwmset))[0]
        pwmnames = np.array(pwmnames)[keep]
        pwms = [pwms[k] for k in keep]
        print('After filter', len(pwmnames))
    
    
    if '--setcut' in sys.argv: # Filter by a value assigned to the pwmname
        print('Before', len(pwmnames))
        pwmset = np.genfromtxt(sys.argv[sys.argv.index('--setcut')+1], dtype = str)
        setcut = float(sys.argv[sys.argv.index('--setcut')+2])
        pwmset = pwmset[pwmset[:,-1].astype(float) >= setcut,0]
        keep = np.where(np.isin(pwmnames, pwmset))[0]
        pwmnames = np.array(pwmnames)[keep]
        pwms = [pwms[k] for k in keep]
        print('After filter', len(pwmnames))

    isEmpty= False
    for p, pwm in enumerate(pwms):
        if np.sum(np.abs(pwm)) == 0:
            print(f'PWM {pwmnames[p]} is empty')
            isEmpty=True
    if isEmpty:
        print('Remove empty PWMs')
        sys.exit()
    
    if '--negatepwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = -pwm
    
    # Modify pwms, consider order of operations
    if '--addpseudo' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] += 0.25
    
    if '--exppwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.exp(pwm)
        
    if '--normpwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.sum(pwm,axis =1)[:, None]
        
    if '--standardpwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.amax(np.sum(pwm,axis =1))
        
    infocont = False
    if '--infocontpwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    if '--infocont' in sys.argv:
        infocont = True

    # Give heatmap to plot next to pwms
    noheatmap = True
    xticklabels = None
    heatmap = None
    vmin, vmax = None, None
    cmap = 'Blues'
    measurex = None
    if '--heatmap' in sys.argv:
        noheatmap, measurex = False, 'euclidean'
        # read in heatmap
        
        start_data = 1# Column where data starts
        if '--start_heatmap' in sys.argv:
            start_data = int(sys.argv[sys.argv.index('--start_heatmap')+1])
        heatnames, xticklabels, heatmap = readalign_matrix_files(sys.argv[sys.argv.index('--heatmap')+1], data_start_column = start_data)
        
        if not np.array_equal(heatnames, pwmnames): # compare the heatmap names to pwms
            sort = [] # sort if necessary
            for p, pwn in enumerate(pwmnames):
                sort.append(list(heatnames).index(pwn))
            heatmap = heatmap[sort]
        vmax = np.amax(np.absolute(heatmap))
        if np.amin(heatmap) < 0:
            cmap = 'coolwarm'
            vmin = -vmax
        else:
            vmin = 0
        print('Heatmap', np.shape(heatmap))
        
        if '--filtermax' in sys.argv: # use the maximum value of each heatmap column for filtering
            top = float(sys.argv[sys.argv.index('--filtermax')+1])
            if top > 1: 
                top = int(top)
            else:
                top = int(len(heatmap)*top)
            if outname is not None:
                outname +='top'+str(top)
            maxheat = np.amax(np.absolute(heatmap), axis =1)
            keep = np.sort(np.argsort(-maxheat)[:top])
            heatmap = heatmap[keep]
            pwmnames = np.array(pwmnames)[keep]
            pwms = [pwms[k] for k in keep]
    
    # provide pwm features for barplots or boxplots
    pwmfeatures = None 
    featurekwargs = {}
    if '--pwmfeatures' in sys.argv:
        featurefile = sys.argv[sys.argv.index('--pwmfeatures')+1]
        if os.path.splitext(featurefile)[-1] == '.npz':
            # readin npz file
            pffile = np.load(featurefile, allow_pickle = True)
            pfnames = pffile['names']
            pfeffects = pffile['effects']
        else:
            pffile = np.genfromtxt(featurefile, dtype = str)
            pfnames = pffile[:,0]
            pfeffects = pffile[:,[1]].astype(float)
            upf, upfN = np.unique(pfnames, return_counts = True)
            if len(np.unique(upfN)) == 1:
                pfnames = pfnames[:int(len(pfnames)/upfN[0])]
        # Join determines how many boxplots are shown for each data point
        if len(pfeffects)%len(pfnames) != 0:
            print('pwmfeatures do not match pfnames')
            sys.exit()
        join = int(len(pfeffects)/len(pfnames)) # if more effects than names in file
        # join will be larger
        pwmfeatures = [[] for j in range(join)]
        keep = [] # sort features to pwmnames and only keep pwm names with features
        for p, pwmname in enumerate(pwmnames):
            for j in range(join):
                if pwmname in pfnames:
                    pwmfeatures[j].append(pfeffects[j *len(pfnames)+list(pfnames).index(pwmname)])
                    if j == 0: 
                        keep.append(p)
        if len(keep) < len(pwmnames):
            notkeep = np.arange(len(pwmnames))
            notkeep = notkeep[~np.isin(notkeep, keep)]
            print('ATTENTION', pwmnames[notkeep], 'have no feature and were removed')
        pwmnames = pwmnames[keep]
        pwms = [pwms[k] for k in keep]
        
        # modify order of features for plot_heatmap function
        pwmfeatures = [m for ma in pwmfeatures for m in ma]
        featurekwargs['split'] = join
        if len(sys.argv) > sys.argv.index('--pwmfeatures')+2:
            # Read in other specifics for box- and barplot with colored swarm
            # Provide all figure specific features as e.g. swarm=True+scattersize=5
            if '=' in sys.argv[sys.argv.index('--pwmfeatures')+2]:
                fkwargs = inputkwargs_from_string(sys.argv[sys.argv.index('--pwmfeatures')+2])
                featurekwargs = featurekwargs | fkwargs
                                        
    # Provide name file and add TF names to cluster names
    yticklabels = np.array(pwmnames, dtype = '<U200')
    if '--pwmnames' in sys.argv:
        # read in txt file 
        nameobj = np.genfromtxt(sys.argv[sys.argv.index('--pwmnames')+1], delimiter = None, dtype = str)
        nnames, repnames = nameobj[:,0], nameobj[:,1]
        assignedpwms = 0
        
        for p, pwmname in enumerate(pwmnames):
            if pwmname in nnames:
                assignedpwms += 1
                yticklabels[p] = repnames[list(nnames).index(pwmname)]
                if pwmfeatures is not None and '--addsizetoname' in sys.argv:
                    yticklabels[p] += '('+str(len(pwmfeatures[p]))+')'
        print('PWMnames assigned', assignedpwms, len(pwms))

    # By default, tree will be build from similarity between motifs
    # Use reverse complem_complement for computing motif similarity
    revcom_array = False
    if '--reverse_complement' in sys.argv:
        revcom_array = True
    min_sim = 4
    if '--min_overlap' in sys.argv:
        min_sim = int(sys.argv[sys.argv.index('--min_overlap')+1])
    # If heat map given, can also use the heatmap to build the tree instead
    if '--heatmap' in sys.argv and '--sortbyheatmap' in sys.argv:
        heatdist = sys.argv[sys.argv.index('--sortbyheatmap')+1]
        # provide preferred distance matrix
        correlation = cdist(heatmap, heatmap, heatdist)
        if outname is not None:
            outname += heatdist[:3]
    else:
        correlation, ofs, revcomp_matrix = torch_compute_similarity_motifs(pwms, pwms, metric = 'correlation', min_sim = min_sim, infocont = False, reverse_complement = revcom_array, return_alignment= True, exact = True)
        # For visualization purposes, it might be better join motifs if they are too similar
        if '--joinpwms' in sys.argv:
            distance_threshold=float(sys.argv[sys.argv.index('--joinpwms')+1]) # define the
            # threshold at which two motifs are combined
            outname += 'joined'+str(distance_threshold)+'pfm'
            newclustering = AgglomerativeClustering(n_clusters = None, metric = 'precomputed', linkage = 'average', distance_threshold = distance_threshold).fit(correlation)
            newclusters = newclustering.labels_
            # combine the pwms for cluster
            
            clusterpwms = combine_pwms(np.array(pwms, dtype = object), newclusters, 1-correlation, ofs, revcomp_matrix)
            # combine the names
            clusternames = [';'.join(np.array(yticklabels)[newclusters == i]) for i in np.unique(newclusters)]
            yticklabels, pwms = np.array(clusternames), clusterpwms
            
            # Compute new relationship
            correlation, ofs, revcomp_matrix = torch_compute_similarity_motifs(pwms, pwms, metric = 'correlation', min_sim = min_sim, infocont = False, reverse_complement = revcom_array, return_alignment = True, exact = True)
            
            if pwmfeatures is not None: # Combine their features as well
                additive = False
                unclusters = np.unique(newclusters)
                newpwmfeatures = []
                # barplot features will be added, while arrays will be concantenated
                if 'barplot' in featurekwargs.keys():
                    if featurekwargs['barplot']:
                        additive = True
                if additive:
                    for u,uc in enumerate(unclusters):
                        newpwmfeatures.append([np.sum(np.array(pwmfeatures)[newclusters == uc])])
                else:
                    for u,uc in enumerate(unclusters):
                        newpwmfeatures.append([np.concatenate(np.array(pwmfeatures)[newclusters == uc])])
                pwmfeatures = newpwmfeatures
            if heatmap is not None:
                unclusters = np.unique(newclusters)
                newheatmap = np.zeros((len(unclusters), heatmap.shape[-1]))
                heatjoin= 'mean'
                if '--heatjoin' in sys.argv:
                    heatjoin = sys.argv[sys.argv.index('--heatjoin')+1]
                for u,uc in enumerate(unclusters):
                    if heatjoin == 'max':
                        argmax = np.argmax(np.abs(heatmap[newclusters == uc]), axis = 0)
                        newheatmap[u] = heatmap[newclusters == uc][(argmax, np.arange(heatmap.shape[-1]))]
                    else:
                        newheatmap[u] = np.mean(heatmap[newclusters == uc], axis = 0)
                
            
            print('Joined to ', len(pwms))
            if '--savejoined' in sys.argv:
                write_meme_file(clusterpwms, clusternames, 'ACGT', outname +'s.meme')
                print(outname +'s.meme')
                if pwmfeatures is not None:
                    np.savez_compressed(outname+'feats.npz', names = clusternames, effects = pwmfeatures)
            
        # adjust the strand for the motifs before plotting
        if '--reverse_complement' in sys.argv:
            best = np.argmin(np.sum(correlation))
            for p, pwm in enumerate(pwms):
                if revcomp_matrix[p,best] == 1:
                    pwms[p] = reverse(pwm)
    
    sortx = 'ward'
    if '--sortx' in sys.argv:
        sortx = check(sys.argv[sys.argv.index('--sortx')+1])
    
    sorty = 'ward'
    if '--sorty' in sys.argv:
        sorty = check(sys.argv[sys.argv.index('--sorty')+1])
    
    if '--measure' in sys.argv:
        measurex = sys.argv[sys.argv.index('--measure')+1]
    
    plot_heatmap(heatmap, # matrix that is plotted with imshow
                 ydistmat = correlation, # matrix that determines order of data points on y-axis
                 measurex = measurex, # if matrix is not a symmetric distance matrix then measurex define distannce metric for linkage clustering 
                 measurey = measurex, # same as measurex just for y axic
                 sortx = sortx, # agglomerative clustering algorith used in likage, f.e average, or single
                 sorty = sorty, # same as above but for y axis
                 x_attributes = None, # additional heatmap with attributes of columns
                 y_attributes = None, # same as above for y axis
                 xattr_name = None, # names of attributes for columns
                 yattr_name = None, # names of attributes for rows
                 heatmapcolor = cmap, # color map of main matrix
                 xatt_color = None, # color map or list of colormaps for attributes
                 yatt_color = None, 
                 xatt_vlim = None, # vmin and vmas for xattributes, or list of vmin and vmax
                 yatt_vlim = None,
                 pwms = pwms, # pwms that are plotted with logomaker next to rows of matrix
                 infocont = infocont,
                 color_cutx = 0., # cut off for coloring in linkage tree. 
                 color_cuty = 0., 
                 xdenline = None, # line drawn into linkage tree on x-axis
                 ydenline = None, 
                 plot_value = False, # if true the values are written into the cells of the matrix
                 vmin = vmin, # min color value 
                 vmax = vmax, 
                 grid = False, # if True, grey grid drawn around heatmap cells
                 xlabel = None, # label on x-axis
                 ylabel = None, # ylabel
                 xticklabels = xticklabels,
                 yticklabels = yticklabels ,
                 showdpi = 100,
                 dpi = dpi,
                 figname = outname,
                 fmt = '.jpg',
                 maxsize = 150, 
                 cellsize = 0.4,
                 cellratio = 1.,
                 noheatmap = noheatmap,
                 row_distributions = pwmfeatures,
                 row_distribution_kwargs = featurekwargs)
