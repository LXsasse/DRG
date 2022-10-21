import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce
import glob
import seaborn as sns
from matplotlib import cm
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

def plot_heatmap(heatmat, measurex = None, measurey = None, sortx = None, sorty = None, x_attributes = None, y_attributes = None, xattr_name = None, yattr_name = None, heatmapcolor = cm.BrBG_r, xatt_color = None, yatt_color = None, pwms = None, combine_cutx = 0., combine_cuty = 0., color_cutx = 0., color_cuty = 0., plot_value = False, vmin = None, vmax = None, grid = False, xdenline = None, ydenline = None, xlabel = None, ylabel = None, xticklabels = None, yticklabels  = None, dpi = 100, figname = None, maxsize = 20):
    
    # either provide similarity matrix as heatmap (measurex = None) or provide a similarity function from scipy.spatial.distance.pdist
    # If no measure is provided heatmap entries will be rescaled between 0,1 and a transformation function can retransform for xticklabels
    if measurex is not None:
        simatrixX = pdist(heatmat.T, metric = measurex)
    else:
        if np.shape(heatmat)[0] != np.shape(heatmat)[1]:
            print( 'heatmat not symmetric matrix: sortx set to None if given')
            sortx = None
        else:
            if np.any(np.abs(heatmat - heatmat.T) > 1e-8):
                print( 'heatmat not symmetric matrix: sortx set to None if given')
                sortx = None
        
        if sortx is not None:        
            # checks if similarity matrix or distance matrix
            issimilarity = np.all(np.amax(heatmat) == np.diag(heatmat))
            heatmax, heatmin = np.amax(heatmat), np.amin(heatmat)
            simatrixX = 1.-heatmat.T #int(issimilarity) - (2.*int(issimilarity)-1.) * (heatmat - heatmin)/(heatmax - heatmin)
            simatrixX = simatrixX[np.triu_indices(len(simatrixX),1)]
            
    if measurey is not None:
        simatrixY = pdist(heatmat, metric = measurey)
    else:
        if np.shape(heatmat)[0] != np.shape(heatmat)[1]:
            print( 'heatmat not symmetric matrix: sorty set to None if given')
            sorty = None
        else:
            if np.any(np.abs(heatmat - heatmat.T) > 1e-8):
                print( 'heatmat not symmetric matrix: sorty set to None if given')
                sorty = None
        if sorty is not None:        
            # checks if similarity matrix or distance matrix
            issimilarity = np.all(np.amax(heatmat) == np.diag(heatmat))
            heatmax, heatmin = np.amax(heatmat), np.amin(heatmat)
            simatrixY = 1.-heatmat #int(issimilarity) - (2.*int(issimilarity)-1.) * (heatmat - heatmin)/(heatmax - heatmin)
            simatrixY = simatrixY[np.triu_indices(len(simatrixY),1)]
            
            
    
    
    # Generate dendrogram for x and y
    #### NJ not yet included
    if sortx is not None:
        Zx = linkage(simatrixX, sortx)
        #if combine_cutx > 0:
            #Zx = reduce_z(Zx, combine_cutx)
    if sorty == 'maxdist':
        sortsize = np.argsort(heatmat[:,0] -heatmat[:,1] + np.amax(heatmat, axis = 1)*0.1)
        
    elif sorty is not None:
        Zy = linkage(simatrixY, sorty) 
        #if combine_cuty > 0:
            #Zy = reduce_z(Zy, combine_cuty)
    xextra = 0.
    if y_attributes is not None:
        xextra = np.shape(y_attributes)[1]*0.8
    yextra = 0.
    if x_attributes is not None:
        yextra = np.shape(x_attributes)[0]*0.8
    

    fig = plt.figure(figsize = (min(maxsize,0.3*np.shape(heatmat)[1])+xextra, min(maxsize, 0.3*np.shape(heatmat)[0])+yextra), dpi = dpi)
    
    if 0.3*np.shape(heatmat)[1] > maxsize:
        xticklabels = None
        plot_value = False
    if 0.3*np.shape(heatmat)[0] > maxsize:
        yticklabels = None
        plot_value = False
    
    
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_position([0.15,0.15,0.7,0.75])
    ax.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
    
    
    if sortx is not None:
        axdenx = fig.add_subplot(711)
        axdenx.spines['top'].set_visible(False)
        axdenx.spines['right'].set_visible(False)
        axdenx.spines['bottom'].set_visible(False)
        axdenx.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdenx.set_position([0.15,0.91,0.7,0.04])
        dnx = dendrogram(Zx, ax = axdenx, no_labels = True, above_threshold_color = 'k', color_threshold = color_cutx, orientation = 'top')
        
        sortx = dnx['leaves']
        heatmat = heatmat[:, sortx]
        if x_attributes is not None:
            x_attributes = x_attributes[:, sortx]
            
        if xticklabels is not None:
            xticklabels = xticklabels[sortx]
            
        if xdenline is not None:
            axdenx.plot([0,len(heatmat[0])*10], [xdenline, xdenline], color = 'r')
    else:
        sortx = np.arange(len(heatmat[0]), dtype = int)
    
    sys.setrecursionlimit(100000)    
    
    if sorty =='maxdist':
        sorty = sortsize
    elif sorty is not None:
        axdeny = fig.add_subplot(171)
        axdeny.spines['top'].set_visible(False)
        axdeny.spines['right'].set_visible(False)
        axdeny.spines['left'].set_visible(False)
        axdeny.tick_params(which = 'both', left = False, labelleft = False)
        axdeny.set_position([0.05,0.15,0.09,0.75])
        dny = dendrogram(Zy, ax = axdeny, no_labels = True, color_threshold = color_cuty, above_threshold_color = 'k', orientation = 'left', get_leaves = True)
        sorty = dny['leaves']
        heatmat = heatmat[sorty]
        #axdeny.set_yticks(axdeny.get_yticks()[1:])

        if y_attributes is not None:
            y_attributes = y_attributes[sorty]
            
        if yticklabels is not None:
            yticklabels = yticklabels[sorty]
        if ydenline is not None:
            axdeny.plot([ydenline, ydenline], [0,len(heatmat)*10], color = 'r')
    else:
        sorty = np.arange(len(heatmat), dtype = int)
    
    
    if vmin is None:
        vmin = np.amin(heatmat)
    if vmax is None:
        vmax = np.amax(heatmat)
    
    ax.imshow(heatmat, aspect = 'auto', cmap = heatmapcolor, vmin = vmin, vmax = vmax, origin = 'lower')
    ax.set_yticks(np.arange(len(heatmat)))
    ax.set_xticks(np.arange(len(heatmat[0])))
    
    if plot_value:
        if np.amax(np.absolute(heatmat)) >= 10:
            heattext = np.array(heatmat, dtype = int)
        else:
            heattext = np.around(heatmat, 2)
        for c in range(len(heattext[0])):
            for d in range(len(heattext)):
                ax.text(c,d,str(heattext[d,c]), color = 'k', ha = 'center', fontsize = 6)
    
    
    if grid:
        ax.set_yticks(np.arange(len(heatmat)+1)-0.5, minor = True)
        ax.set_xticks(np.arange(len(heatmat[0])+1)-0.5, minor = True)
        ax.grid(color = 'k', which = 'minor')


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
        axatx.set_position([0.15,0.11,0.7,0.04])
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
        ax.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
        ax.set_xticklabels(xticklabels, rotation = 90)
        
    
    if y_attributes is not None:
        for y, yunique in enumerate(y_attributes.T):
            yunique = np.unique(yunique)
            for s, yuni in enumerate(yunique):
                y_attributes[y_attributes[:,y] == yuni,y] = s
        y_attributes = y_attributes.astype(int)
        axaty = fig.add_subplot(177)
        axaty.spines['top'].set_visible(False)
        axaty.spines['bottom'].set_visible(False)
        axaty.spines['right'].set_visible(False)
        axaty.spines['left'].set_visible(False)
        axaty.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
        axaty.set_position([0.85,0.15,0.03,0.75])
        axaty.imshow(y_attributes, aspect = 'auto', cmap = yatt_color)
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
        ax.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticks(np.arange(0, len(heatmat),200))
        ax.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
        
    

    if figname is not None:
        fig.savefig(figname+'_heatmap.jpg', bbox_inches = 'tight')
        print( 'SAVED', figname)
    else:
        plt.show()
    plt.close()
    return sortx, sorty


def plot_distribution(matrix, modnames, outname = None, xwidth = 0.6, height = 4, width = 0.8, show_mean = True, grid = True, swarm = True, plotnames = 0, datanames = None, scatter_color = None, scatter_colormap = cm.jet, scatter_alpha = 0.8, scatter_size = 0.5, sort = 'top', sizemax = 2, sizemin = 0.25, colormin = None, colormax = None, dpi = 200, savedpi = 200, xorder = 'size', ylabel = None):
    fig = plt.figure(figsize = (len(modnames)*xwidth, height), dpi = dpi)
    ax = fig.add_subplot(111)
    ax.set_position([0.1,0.1,0.8,0.8])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    matrix = list(matrix)
    if swarm:
        #sns.swarmplot(data=matrix, color = 'k', size = size, ax = ax)
        # replace swarmplot with scatterplot with random location within width
        if colormin is None:
            colormin = np.amin(scatter_color)
        if colormax is None:
            colormax = np.amax(scatter_color)
        
        for i, set1 in enumerate(matrix):
            set1 = np.array(set1)
            if sort == 'top':
                setsort = np.argsort(set1)
            else:
                setsort = np.argsort(-set1)
            
            if scatter_color is None:
                scatter_colormap = cm.twilight
                sccolor = np.ones(len(setsort))*0.25
            else:
                sccolor = (scatter_color-colormin)/(colormax-colormin)
                
            if scatter_size is None:
                scsize = np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            elif isinstance(scatter_size, float) or isinstance(scatter_size, int):
                scsize = scatter_size * np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            else:
                scsize = np.sqrt(scatter_size/3.)
                scsize = (((sizemax-sizemin)*(scsize - np.amin(scsize))/(np.amax(scsize) - np.amin(scsize))) + sizemin)
                scsize *= plt.rcParams['lines.markersize'] ** 2.
                
            if xorder == 'size' and scatter_size is not None and not isinstance(scatter_size, float) and not isinstance(scatter_size, int):
                randx = i + width * ((scsize-np.amin(scsize))/(np.amax(scsize)-np.amin(scsize)) - 0.5)
            else:
                randx = i + width * (np.random.random(len(setsort))-0.5)
            
            ax.scatter(randx[setsort], set1[setsort], cmap= scatter_colormap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0)
                
        # Determine which scatters should get a name written next to them
        if plotnames> 0 and datanames is not None:
            for mat in matrix:
                which_to_plot = np.argsort(-mat)[:plotnames]
    
    if scatter_color is not None and swarm:
        axcol = fig.add_subplot(911)
        axcol.set_position([0.6,0.925,0.3, 0.05])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = scatter_colormap)
        axcol.set_xticks([0,50,101])
        axcol.set_xticklabels([round(colormin,1), round((colormin+colormax)/2,1), round(colormax,1)], rotation = 90)
    
    ax = sns.boxplot(data=matrix,showcaps=False,boxprops={'facecolor':'None'},
    showfliers=False,whiskerprops={'linewidth':1},ax = ax, width = width)
    if show_mean:
        ax.plot(np.arange(len(matrix)), [np.mean(mat) for mat in matrix], color = 'r', marker = 's')
    ax.set_xticks(np.arange(len(modnames)))
    ax.set_xticklabels(modnames, rotation = 90)
    if grid:
        ax.grid(axis = 'y')
    if outname is None:
        #fig.tight_layout()
        plt.show()
    else:
        fig.savefig(outname+'_distribution.jpg', dpi = savedpi, bbox_inches = 'tight')


