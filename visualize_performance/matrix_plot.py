import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce
import glob
from matplotlib import cm
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib as mpl

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


def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def approximate_density(set1, bins = 20, moves = 4, miny=None, maxy = None):
    if miny is None:
        miny = np.amin(set1)
    if maxy is None:
        maxy = np.amax(set1)
    bsize = (maxy-miny)/bins
    for m in range(moves):
        bins1 = np.linspace(miny-(m+1)*bsize/(moves+1), maxy-(m+1)*bsize/(moves+1),bins + 1)
        bins2 = np.linspace(miny+(m+1)*bsize/(moves+1), maxy+(m+1)*bsize/(moves+1),bins + 1)
        density= np.array([np.sum((set1 >= bins1[b]) * (set1<bins1[b+1])) for b in range(len(bins1)-1)])
        density2= np.array([np.sum((set1 >= bins2[b]) * (set1<bins2[b+1])) for b in range(len(bins2)-1)])
        density = density/np.amax(density)
        density2 = density2/np.amax(density2)
        dens = np.zeros(len(set1))
        dcount = np.zeros(len(set1))
        for b in range(bins):
            dens[(set1 >= bins1[b]) * (set1<bins1[b+1])] += density[b]
            dens[(set1 >= bins2[b]) * (set1<bins2[b+1])] += density[b]
            dcount[(set1 >= bins1[b]) * (set1<bins1[b+1])] += 1
            dcount[(set1 >= bins2[b]) * (set1<bins2[b+1])] += 1
        dens = dens/dcount
    return dens

def plot_distribution(matrix, modnames, split = 1, outname = None, xwidth = 0.6, height = 4, width = 0.8, show_mean = False, showfliers = False, showcaps = True, facecolor = None, grid = True, swarm = False, plotnames = 0, datanames = None, scatter_color = 'grey', scatter_colormap = cm.jet, scatter_alpha = 0.8, scatter_size = 0.5, connect_swarm = True, sort = 'top', sizemax = 2, sizemin = 0.25, colormin = None, colormax = None, dpi = 200, savedpi = 200, xorder = 'size', ylabel = None, fmt = 'jpg'):
    
    positions = np.arange(len(matrix))
    print(positions)
    fcolor = None
    if split > 1:
        if len(matrix) == split:
            matrix = [m for ma in matrix for m in ma]
        
        if width * split >1:
            width = width/split
        
        positions = []
        for s in range(split):
            positions.append(np.arange(int(len(matrix)/split)) + width*s - (split*width/2) + width/2)
        positions = np.concatenate(positions)
        
        if isinstance(facecolor, list):
            if len(facecolor) == split:
                fcolor = [facecolor[c] for c in range(split) for j in range(int(len(matrix)/split))]
            else:
                fcolor = [facecolor[c] for c in range(len(matrix))]
            facecolor = None
            
    
    if facecolor is None:
        facecolor = (0,0,0,0)

    
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
        
        if colormin is None and not isinstance(scatter_color, str):
            colormin = np.amin(scatter_color)
        if colormax is None and not isinstance(scatter_color, str):
            colormax = np.amax(scatter_color)
        
        if connect_swarm and len(np.shape(matrix)) > 1:
            xposses = []
            randomshift = np.random.random(len(matrix[0]))
        
        for i, set1 in enumerate(matrix):
            set1 = np.array(set1)
            if sort == 'top':
                setsort = np.argsort(set1)
            else:
                setsort = np.argsort(-set1)
            
            if scatter_color is None:
                scatter_colormap = cm.twilight
                sccolor = np.ones(len(setsort))*0.25
            elif isinstance(scatter_color, str):
                sccolor = np.array([scatter_color for ci in range(len(setsort))])
            else:
                sccolor = (scatter_color-colormin)/(colormax-colormin)
                
            if scatter_size is None:
                scsize = 0.2*np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            elif isinstance(scatter_size, float) or isinstance(scatter_size, int):
                scsize = scatter_size * np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
            else:
                scsize = np.sqrt(scatter_size/3.)
                scsize = (((sizemax-sizemin)*(scsize - np.amin(scsize))/(np.amax(scsize) - np.amin(scsize))) + sizemin)
                scsize *= plt.rcParams['lines.markersize'] ** 2.
                
            if xorder == 'size' and scatter_size is not None and not isinstance(scatter_size, float) and not isinstance(scatter_size, int):
                randx = positions[i] + width * ((scsize-np.amin(scsize))/(np.amax(scsize)-np.amin(scsize)) - 0.5)
            else:
                dens = approximate_density(set1)
                if connect_swarm and len(np.shape(matrix)) > 1:
                    randx = positions[i] + dens *width/2 * (randomshift-0.5)
                else:
                    randx = positions[i] + dens * width * (np.random.random(len(setsort))-0.5) # + width/2 * simple_beeswarm(set1, nbins = 40) #
            
            ax.scatter(randx[setsort], set1[setsort], cmap= scatter_colormap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
            if connect_swarm and len(np.shape(matrix)) > 1:
                xposses.append(randx)
        
        if connect_swarm and len(np.shape(matrix)) > 1:
            xposses=np.array(xposses)
            for j, setj in enumerate(np.array(matrix).T):
                ax.plot(xposses[:,j], setj, color  = 'grey', alpha = 0.5, lw = 0.5)
            
        # Determine which scatters should get a name written next to them
        if plotnames> 0 and datanames is not None:
            for mat in matrix:
                which_to_plot = np.argsort(-mat)[:plotnames]
    # generate colorbar
    if ((scatter_color is not None) and (not isinstance(scatter_color, str))) and swarm:
        axcol = fig.add_subplot(911)
        axcol.set_position([0.6,0.925,0.3, 0.05])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = scatter_colormap)
        axcol.set_xticks([0,50,101])
        axcol.set_xticklabels([round(colormin,1), round((colormin+colormax)/2,1), round(colormax,1)], rotation = 90)
    
    bplot = ax.boxplot(matrix, positions = positions, showcaps=showcaps, patch_artist = True, boxprops={'facecolor':facecolor}, showfliers=showfliers, whiskerprops={'linewidth':1}, widths = width,zorder = 4)
    
    if fcolor is not None:
        for patch, color in zip(bplot['boxes'], fcolor):
            patch.set_facecolor(color)
            fc = patch.get_facecolor()
            patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
            
    
    if show_mean:
        ax.plot(np.sort(positions), [np.mean(matrix[s]) for s in np.argsort(positions)], color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
    
    ax.set_xticks(np.arange(len(modnames)))
    ax.set_xticklabels(modnames, rotation = 90)
    if grid:
        ax.grid(axis = 'y')
    if outname is None:
        #fig.tight_layout()
        plt.show()
    else:
        fig.savefig(outname+'_distribution.'+fmt, dpi = savedpi, bbox_inches = 'tight')


