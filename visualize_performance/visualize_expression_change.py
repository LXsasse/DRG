import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import logomaker as lm
from scipy.spatial.distance import pdist, cdist
import matplotlib.cm as cm
# plots heatmap with preferred colormap
# sorts by dedrogram or tree agglomerative clustering
# allows to insert functional annotations at x and y and provide color-codes for that
# can simplify tree over a certain value
# plots the actual value of the value to a given accuracy 0,1,2,..

def reduce_z(Z, cutoff):
    for i,z in enumerate(Z):
        if z[2] < cutoff:
            Z[i,2] = 0
    return Z




def plot_heatmap(heatmat, measurex = None, measurey = None, sortx = None, sorty = None, x_attributes = None, y_attributes = None, sorty_attributes = False, sortx_attributes = False, xattr_name = None, yattr_name = None, heatmapcolor = cm.BrBG, xatt_color = None, yatt_color = None, pwms = None, combine_cutx = 0., combine_cuty = 0., color_cutx = 0., color_cuty = 0., plot_value = False, vmin = None, vmax = None, grid = False, xdenline = None, ydenline = None, xlabel = None, ylabel = None, xticklabels = None, yticklabels  = None, dpi = 100, figname = None):
    
    # either provide similarity matrix as heatmap (measurex = None) or provide a similarity function from scipy.spatial.distance.pdist
    # If no measure is provided heatmap entries will be rescaled between 0,1 and a transformation function can retransform for xticklabels
    if x_attributes is not None and sortx_attributes:
        simatrixX = pdist(np.array(x_attributes), metric = measurex)
        def transformx(treevals):
            return treevals
        
    elif measurex is not None:
        simatrixX = pdist(heatmat.T, metric = measurex)
        def transformx(treevals):
            return treevals
    else:
        if np.shape(heatmat)[0] != np.shape(heatmat)[1]:
            print 'heatmat not symmetric matrix: sortx set to None if given'
            sortx = None
        else:
            if np.all(np.abs(heatmat - heatmat.T) > 10^-8):
                print 'heatmat not symmetric matrix: sortx set to None if given'
                sortx = None
        if sortx is not None:        
            # checks if similarity matrix or distance matrix
            issimilarity = np.amax(heatmap) == np.diag(heatmap)
            maxheat, minheat = np.amax(heatmat), np.amin(heatmat)
            def transformx(treevals):
                return (treevals - int(issimilarity))*- (2.*int(issimilarity)-1.) * (heatmax - heatmin)+heatmin
            simatrixX = int(issimilarity) - (2.*int(issimilarity)-1.) * (heatmat - heatmin)/(heatmax - heatmin)
            simatrixX = simatrixX[np.triu_indices(len(simatrixX),1)]
    
    
    if y_attributes is not None and sorty_attributes:
        simatrixY = pdist(np.array(y_attributes), metric = measurey)
        def transformy(treevals):
            return treevals
        
    elif measurey is not None:
        simatrixY = pdist(heatmat, metric = measurey)
        
        def transformy(treevals):
            return treevals
    else:
        if np.shape(heatmat)[0] != np.shape(heatmat)[1]:
            print 'heatmat not symmetric matrix: sorty set to None if given'
            sorty = None
        else:
            if np.all(np.abs(heatmat - heatmat.T) > 10^-8):
                print 'heatmat not symmetric matrix: sortx set to None if given'
                sorty = None
        if sorty is not None:        
            # checks if similarity matrix or distance matrix
            issimilarity = np.amax(heatmap) == np.diag(heatmap)
            maxheat, minheat = np.amax(heatmat), np.amin(heatmat)
            def transformx(treevals):
                return (treevals - int(issimilarity))*- (2.*int(issimilarity)-1.) * (heatmax - heatmin)+heatmin
            simatrixY = int(issimilarity) - (2.*int(issimilarity)-1.) * (heatmat - heatmin)/(heatmax - heatmin)
            simatrixY = simatrixY[np.triu_indices(len(simatrixY),1)]
    
    
    # Generate dendrogram for x and y
    #### NJ not yet included
    if sortx is not None:
        Zx = linkage(simatrixX, sortx)
        #if combine_cutx > 0:
            #Zx = reduce_z(Zx, combine_cutx)
    if sorty is not None:
        Zy = linkage(simatrixY, sorty) 
        #if combine_cuty > 0:
            #Zy = reduce_z(Zy, combine_cuty)
    xextra = 0.

    if y_attributes is not None:
        xextra = np.shape(y_attributes)[1]*0.8
    yextra = 0.
    if x_attributes is not None:
        yextra = np.shape(x_attributes)[0]*0.8

    fig = plt.figure(figsize = (min(3000/dpi,0.3*np.shape(heatmat)[1])+xextra, min(3000/dpi, 0.3*np.shape(heatmat)[0])+yextra), dpi = dpi)

    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_position([0.15,0.15,0.7,0.7])
    ax.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
    
    
    if sortx is not None:
        axdenx = fig.add_subplot(711)
        axdenx.spines['top'].set_visible(False)
        axdenx.spines['right'].set_visible(False)
        axdenx.spines['bottom'].set_visible(False)
        axdenx.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdenx.set_position([0.15,0.85,0.7,0.1])
        dnx = dendrogram(Zx, ax = axdenx, no_labels = True, above_threshold_color = 'k', color_threshold = color_cutx, orientation = 'top')
        
        sortx = dnx['leaves']
        heatmat = heatmat[:, sortx]
        if x_attributes is not None:
            print x_attributes
            x_attributes = x_attributes[:, sortx]
            print x_attributes
            
        if xticklabels is not None:
            xticklabels = xticklabels[sortx]
            print xticklabels
        if xdenline is not None:
            axdenx.plot([0,len(heatmat[0])*10], [xdenline, xdenline], color = 'r')
    
    
    sys.setrecursionlimit(100000)    
    if sorty is not None:
        axdeny = fig.add_subplot(171)
        axdeny.spines['top'].set_visible(False)
        axdeny.spines['right'].set_visible(False)
        axdeny.spines['left'].set_visible(False)
        axdeny.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdeny.set_position([0.05,0.15,0.1,0.7])
        dny = dendrogram(Zy, ax = axdeny, no_labels = True, color_threshold = color_cuty, above_threshold_color = 'k', orientation = 'left', get_leaves = True)
        sorty = dny['leaves']
        heatmat = heatmat[sorty]

        if y_attributes is not None:
            y_attributes = y_attributes[sorty]
        if yticklabels is not None:
            yticklabels = yticklabels[sorty]
        if ydenline is not None:
            axdeny.plot([ydenline, ydenline], [0,len(heatmat)*10], color = 'r')
    
    if vmin is None:
        vmin = np.amin(heatmat)
    if vmax is None:
        vmax = np.amax(heatmat)
    
    ax.imshow(heatmat, aspect = 'auto', cmap = heatmapcolor, vmin = vmin, vmax = vmax)
    ax.set_yticks(np.arange(len(heatmat)))
    ax.set_xticks(np.arange(len(heatmat[0])))
    
    if grid:
        ax.set_yticks(np.arange(len(heatmat)+1)-0.5, which = 'minor')
        ax.set_xticks(np.arange(len(heatmat[0])+1)-0.5, which = 'minor')
        ax.grid(color = 'k')


    if x_attributes is not None:
        for x, xunique in enumerate(x_attributes):
            xunique = np.unique(xunique)
            for s, xuni in enumerate(xunique):
                x_attributes[x, x_attributes[x] == xuni] = s
        x_attributes = x_attributes.astype(int)
        axatx = fig.add_subplot(717)
        axatx.spines['top'].set_visible(False)
        axatx.spines['bottom'].set_visible(False)
        axatx.spines['right'].set_visible(False)
        axatx.spines['left'].set_visible(False)
        axatx.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False, labelright = True)
        axatx.set_position([0.15,0.05,0.7,0.1])
        axatx.imshow(x_attributes, aspect = 'auto', cmap = xatt_color)
        axatx.set_xticks(np.arange(len(heatmat[0])))        
        if xlabel is not None:
            axatx.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
            axatx.set_xlabel(xlabel)
        if xticklabels is not None:
            axatx.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
            axatx.set_xticklabels(xticklabels, rotation  = 90)
    elif xlabel is not None:
        ax.set_xlabel(xlabel)
    elif xticklabels is not None:
        ax.set_xticklabels(xticklabels)
        
    
    if y_attributes is not None:
        axaty = fig.add_subplot(177)
        axaty.spines['top'].set_visible(False)
        axaty.spines['bottom'].set_visible(False)
        axaty.spines['right'].set_visible(False)
        axaty.spines['left'].set_visible(False)
        axaty.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
        axaty.set_position([0.85,0.15,0.1,0.7])
        axaty.imshow(y_attributes, aspect = 'auto', cmap = yatt_color, vmin = -1, vmax =1)
        axaty.set_yticks(np.arange(len(heatmat)))
        if ylabel is not None:
            axaty.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
            axaty.set_ylabel(ylabel)
        if yticklabels is not None:
            axaty.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True, right = True)
            axaty.set_yticklabels(yticklabels)
    elif ylabel is not None:
        ax.set_ylabel(ylabel)
    elif yticklabels is not None:
        ax.set_yticklabels(yticklabels)
  
    

    if figname is not None:
        fig.savefig(figname, bbox_inches = 'tight')
        print 'SAVED', figname
    else:
        fig.tight_layout()
        plt.show()
    plt.close()
    return



def read_in_expression(datafile, delimiter = ','):
    dataread = open(datafile, 'r').readlines()
    experiments = dataread[0].strip().split(delimiter)[2:-1]
    gene_features = [] #dataread[1:][:, 1].astype(int)
    genenames = [] #dataread[1:][:, 1]
    foldchanges = [] #dataread[1:][2:-1].astype(float)
    exp_features = []
    for exp in experiments:
        if 'PBS' in exp:
            split = 'PBS'
        elif '.IL' in exp:
            split = 'IL'
        exp_features.append([exp.split(split, 1)[0], split+exp.split(split, 1)[1].split('.',1)[0]])
    exp_features = np.array(exp_features).T
    
    for l, line in enumerate(dataread):
        if l > 0:
            line = line.replace(',,',',0,0')
            line = line.replace('0-', '-')
            line = line.strip().split(',')
            gene_features.append([int(line[1])])
            genenames.append(line[0])
            foldchanges.append(np.array(line[2:-1], dtype = float))
        
    return np.array(genenames), np.array(experiments), np.array(gene_features), np.array(exp_features), np.array(foldchanges)

def read_in_foldchange(datafile, delimiter = ',', log2 = False):
    dataread = open(datafile, 'r').readlines()
    experiments = dataread[0][1:].strip().split(delimiter)
    gene_features = None 
    genenames = [] 
    foldchanges = [] 
    exp_features = []
    for exp in experiments:
        exp_features.append([exp.split('.', 1)[0], exp.split('.', 1)[1]])
    exp_features = np.array(exp_features).T
    
    for l, line in enumerate(dataread):
        if l > 0:
            line = line.strip().split(',')
            genenames.append(line[0])
            if log2:
                foldchanges.append(np.log2(np.array(line[1:], dtype = float)))
            else:
                foldchanges.append(np.array(line[1:], dtype = float))
            #print foldchanges[-1]
    foldchanges = np.array(foldchanges)
    genenames = np.array(genenames)
    experiments = np.array(experiments)
    foldmasky = np.sum(foldchanges**2, axis = 1) != 0
    genenames, foldchanges = genenames[foldmasky], foldchanges[foldmasky]
    foldmaskx = np.sum(foldchanges**2, axis = 0) != 0
    experiments, foldchanges, exp_features = experiments[foldmaskx], foldchanges[:, foldmaskx], exp_features[:, foldmaskx]
    return genenames, experiments, None, np.array(exp_features), foldchanges


if __name__ == '__main__':
    
    infile = sys.argv[1]
    genenames, experiments, gene_features, exp_features, foldchanges = read_in_foldchange(infile, log2=False)
    print len(genenames), len(experiments), gene_features, np.shape(exp_features), np.shape(foldchanges)
    #### USE color map that colors everything that is significant 
    outname = os.path.splitext(infile)[0]
    if '--filter_genes' in sys.argv:
        genelistfile = sys.argv[sys.argv.index('--filter_genes')+1]
        genelist = np.genfromtxt(genelistfile, dtype = str)
        genemask = np.isin(genenames, genelist)
        print 'Kept', np.sum(genemask), 'significant genes of', len(genenames), 'genes'
        genenames, foldchanges = genenames[genemask], foldchanges[genemask]
        outname += '_'+os.path.splitext(os.path.split(genelistfile)[1])[0]
    
    elif '--gene_features' in sys.argv:
        genelistfile = sys.argv[sys.argv.index('--gene_features')+1]
        genelist = np.genfromtxt(genelistfile, dtype = str)
        genelist, gene_features = genelist[:,0], genelist[:,1:].astype(float)
        genemask = np.argsort(genenames)[np.isin(np.sort(genenames), genelist)]
        print 'Kept', len(genemask), 'significant genes of', len(genenames), 'genes'
        genenames, foldchanges = genenames[genemask], foldchanges[genemask]
        genemask = np.argsort(genelist)[np.isin(np.sort(genelist), genenames)]
        gene_features, genelist = gene_features[genemask], genelist[genemask]
        
        print(np.array_equal(genenames, genelist))
        print(gene_features)
        outname += '_'+os.path.splitext(os.path.split(genelistfile)[1])[0]
        print(outname)
    
    if '--log_data' in sys.argv:
        foldchanges = np.log(foldchanges)
        outname += '_logdata'
    if '--zscore_data' in sys.argv:
        outname += '_zscore'
        foldchanges -= np.mean(foldchanges,axis= 1)[:, None]
        foldchanges /= np.std(foldchanges,axis= 1)[:, None]
    if '--logmedian_data' in sys.argv:
        foldchanges = np.log2(foldchanges/np.median(foldchanges,axis= 1)[:, None])
        outname += '_logmedian'
    if '--logmean_data' in sys.argv:
        foldchanges = np.log2(foldchanges/np.mean(foldchanges,axis= 1)[:, None]) 
        outname += '_logmean'
    
    if '--remove_quantile' in sys.argv:
        foldchanges = np.nan_to_num(foldchanges)
        quant = np.quantile(np.absolute(foldchanges.flatten()), float(sys.argv[sys.argv.index('--remove_quantile')+1]))
        print 'Remove quantile', quant, len(genenames)
        genemask = np.sum(np.absolute(foldchanges) >= quant,axis = 1) == 0
        genenames, foldchanges = genenames[genemask], foldchanges[genemask]
        print len(genenames)

    print np.amin(foldchanges), np.amax(foldchanges)
    ### make additional figure for feature legend
    plot_heatmap(foldchanges, measurex = 'euclidean', measurey = 'euclidean', sortx = 'single', sorty = 'average', x_attributes = exp_features, y_attributes = gene_features, sorty_attributes= False, heatmapcolor = cm.RdBu, xatt_color = cm.tab20, yatt_color = cm.BrBG, pwms = None, combine_cutx = 0., combine_cuty = 0., color_cutx = 0., color_cuty = 0., plot_value = False, vmin = -2, vmax = 2, grid = None, xdenline = 0., ydenline = 0., xticklabels = experiments, yticklabels = genenames, dpi = 100, figname = outname+'.jpg')
    
    
    



