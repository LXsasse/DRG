
'''
plots a treee with PWMs and if given also a matrix, boxplots, or barplots with the pwms
'''

import numpy as np
import sys, os
import matplotlib.pyplot as plt 

from drg_tools.plotlib import plot_heatmap
from drg_tools.motif_analysis import align_compute_similarity_motifs, reverse, combine_pwms
from drg_tools.io_utils import readin_motif_files, write_pwm, check

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':
    
    pwmfile = sys.argv[1]
    outname = os.path.splitext(pwmfile)[0]
    infmt= os.path.splitext(pwmfile)[1]
    nameline = 'Motif'
    
    if '--nameline' in sys.argv:
        nameline = sys.argv[sys.argv.index('--nameline')+1]
        
    pwms,pwmnames,nts = readin_motif_files(pwmfile)
    
    min_sim = 4
    
    if '--usepwmid' in sys.argv:
        pwmnames = np.arange(len(pwmnames)).astype(str)
    elif '--clipname' in sys.argv:
        clip = sys.argv[sys.argv.index('--clipname')+1]
        pwmnames = np.array([pwmname.replace(clip, '') for pwmname in pwmnames])
    
    if '--set' in sys.argv:
        print('Before', len(pwmnames))
        pwmset = np.genfromtxt(sys.argv[sys.argv.index('--set')+1], dtype = str)
        keep = np.where(np.isin(pwmnames, pwmset))[0]
        pwmnames = np.array(pwmnames)[keep]
        pwms = [pwms[k] for k in keep]
        print('After filter', len(pwmnames))
    
    if '--setcut' in sys.argv:
        print('Before', len(pwmnames))
        pwmset = np.genfromtxt(sys.argv[sys.argv.index('--setcut')+1], dtype = str)
        setcut = float(sys.argv[sys.argv.index('--setcut')+2])
        pwmset = pwmset[pwmset[:,-1].astype(float) >= setcut,0]
        keep = np.where(np.isin(pwmnames, pwmset))[0]
        pwmnames = np.array(pwmnames)[keep]
        pwms = [pwms[k] for k in keep]
        print('After filter', len(pwmnames))
        
    
    
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
        
    infocont = True
    if '--infocont' in sys.argv:
        infocont = False
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        dpi = 250
    else:
        outname = None
        dpi = None
    
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
    
    noheatmap = True
    xticklabels = None
    heatmap = None
    vmin, vmax = None, None
    cmap = 'Blues'
    measurex = None
    if '--heatmap' in sys.argv:
        noheatmap, measurex = False, 'euclidean'
        heatmap = np.genfromtxt(sys.argv[sys.argv.index('--heatmap')+1], dtype = str)
        heatnames = heatmap[:,0]
        heatmap = heatmap[:, 2:].astype(float)
        xticklabels = np.array(open(sys.argv[sys.argv.index('--heatmap')+1], 'r').readline().strip('#').strip().split())[2:]
        if not np.array_equal(heatnames, pwmnames):
            sort = []
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
        
        if '--filtermax' in sys.argv:
            top = float(sys.argv[sys.argv.index('--filtermax')+1])
            if top > 1: 
                top = int(top)
            else:
                top = int(len(heatmap)*top)
            outname +='top'+str(top)
            maxheat = np.amax(np.absolute(heatmap), axis =1)
            keep = np.sort(np.argsort(-maxheat)[:top])
            heatmap = heatmap[keep]
            pwmnames = np.array(pwmnames)[keep]
            pwms = [pwms[k] for k in keep]
    
    pwmfeatures = None
    featurekwargs = {}
    if '--pwmfeatures' in sys.argv:
        featurefile = sys.argv[sys.argv.index('--pwmfeatures')+1]
        if os.path.splitext(featurefile)[-1] == '.npz':
            # readin npz fill_logp_self
            pffile = np.load(featurefile, allow_pickle = True)
            pfnames = pffile['names']
            pfeffects = pffile['effects']
        else:
            pffile = np.genfromtxt(featurefile, dtype = str)
            pfnames = pffile[:,0]
            pfeffects = pffile[:,[1]].astype(float)
        
        join = int(len(pfeffects)/len(pfnames))
        print(len(pfeffects), len(pfnames), join)
        print(len(pwmnames), len(pwms))
        pwmfeatures = [[] for j in range(join)]
        keep = []
        for p, pwmname in enumerate(pwmnames):
            for j in range(join):
                if pwmname in pfnames:
                    pwmfeatures[j].append(pfeffects[j *len(pfnames)+list(pfnames).index(pwmname)])
                    #print(pwmname, j, np.mean(pwmfeatures[j][-1]))
                    if j == 0: 
                        keep.append(p)
        pwmnames = np.array(pwmnames)[keep]
        pwms = [pwms[k] for k in keep]
        print(len(pwmfeatures[0]), len(pwmnames), len(pwms))
        pwmfeatures = [m for ma in pwmfeatures for m in ma]
        featurekwargs['split'] = join
        if len(sys.argv) > sys.argv.index('--pwmfeatures')+2:
            if '=' in sys.argv[sys.argv.index('--pwmfeatures')+2]:
                fkwargs = sys.argv[sys.argv.index('--pwmfeatures')+2]
                if '+' in fkwargs:
                    fkwargs = fkwargs.split('+')
                else:
                    fkwargs = [fkwargs]
                for fk in fkwargs:
                    fk = fk.split('=')
                    featurekwargs[fk[0]] = check(fk[1])
                                        
    
    yticklabels = np.array(pwmnames, dtype = '<U200')
    if '--pwmnames' in sys.argv:
        # read in txt file 
        nameobj = np.genfromtxt(sys.argv[sys.argv.index('--pwmnames')+1], delimiter = None, dtype = str)
        nnames, repnames = nameobj[:,0], nameobj[:,1]
        assignedpwms = 0
        #print(nnames, repnames, pwmnames)
        for p, pwmname in enumerate(pwmnames):
            if pwmname in nnames:
                assignedpwms += 1
                yticklabels[p] = repnames[list(nnames).index(pwmname)]
                if pwmfeatures is not None and '--addsizetoname' in sys.argv:
                    yticklabels[p] += '('+str(len(pwmfeatures[p]))+')'
        print('PWMnames assigned', assignedpwms, len(pwms))
        #print(pwmname, yticklabels[p])
        #sys.exit()
    
    revcom_array = np.zeros(len(pwms), dtype = int)
    if '--reverse_complement' in sys.argv:
        revcom_array = np.ones(len(pwms), dtype = int)
    
    if '--heatmap' in sys.argv and '--sortbyheatmap' in sys.argv:
        heatdist = sys.argv[sys.argv.index('--sortbyheatmap')+1]
        correlation = cdist(heatmap, heatmap, heatdist)
        outname += heatdist[:3]
    else:
        njobs = os.cpu_count()
        if '--njobs' in sys.argv:
            njobs = int(sys.argv[sys.argv.index('--njobs')+1])
        correlation, logs, ofs, revcomp_matrix = align_compute_similarity_motifs(pwms, pwms, fill_logp_self = 1000, min_sim = min_sim, infocont = False, reverse_complement = revcom_array, njobs = njobs)
        if '--joinpwms' in sys.argv:
            distance_threshold=float(sys.argv[sys.argv.index('--joinpwms')+1])
            newclustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', linkage = 'average', distance_threshold = distance_threshold).fit(correlation)
            newclusters = newclustering.labels_
            clusterpwms = combine_pwms(np.array(pwms, dtype = object), newclusters, logs, ofs, revcomp_matrix)
            clusternames = [';'.join(np.array(yticklabels)[newclusters == i]) for i in np.unique(newclusters)]
            yticklabels, pwms = np.array(clusternames), clusterpwms
            correlation, logs, ofs, revcomp_matrix = align_compute_similarity_motifs(pwms, pwms, fill_logp_self = 1000, min_sim = min_sim, infocont = False, reverse_complement = revcom_array)
            if pwmfeatures is not None:
                additive = False
                unclusters = np.unique(newclusters)
                newpwmfeatures = []
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
                
            print('Joined to ', len(pwms))
            if '--savejoined' in sys.argv:
                write_pwm(outname +'joined'+str(distance_threshold)+'pfms.txt', clusterpwms, clusternames)
                print(outname +'joined'+str(distance_threshold)+'pfms.txt')
                if pwmfeatures is not None:
                    np.savez_compressed(outname+'joined'+str(distance_threshold)+'pfmfeats.npz', names = clusternames, effects = pwmfeatures)
            
                
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
                 ydistmat = correlation,
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
