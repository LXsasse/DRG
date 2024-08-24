'''
Fancy scatter plots with lots of features
TODO
Switch to drg_tools.plotlib scatterPlot
place wrap some of the larger paragraphs in functions and place on top of script
'''
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde, pearsonr
from matplotlib.colors import ListedColormap

from drg_tools.io_utils import read_matrix_file, isint


def find_in_list(target, thelist):
    for l, lo in enumerate(thelist):
        if target in lo or lo in target:
            return l


def get_index(col, header):
    if isint(col):
        col = int(col)
    elif header is not None:
        try:
            col = list(header).index(col)
        except:
            raise ValueError(f'{col} not a valid column')
    else:
        raise ValueError(f'{col} cannot be transformed to integer')
    return col
    

if __name__ == '__main__':    
    
    file1 = sys.argv[1] # File name with data for x axis
    file2 = sys.argv[2] # File name with data for y axis
    namex = sys.argv[3] # x axis label
    namey = sys.argv[4] # y axis label
    
    # Define delimiter for individual files
    delimiter1 = delimiter2 = None
    if '--delimiter' in sys.argv:
        delimiter1 = delimiter2 = sys.argv[sys.argv.index('--delimiter')+1]
    if '--delimiter1' in sys.argv:
        delimiter1 = sys.argv[sys.argv.index('--delimiter1')+1]
    if '--delimiter2' in sys.argv:
        delimiter2 = sys.argv[sys.argv.index('--delimiter2')+1]
    
    # Read in matrices with data
    names1, header1, vals1 = read_matrix_file(file1, delimiter = delimiter1)
    names2, header2, vals2 = read_matrix_file(file2, delimiter = delimiter2)
    
    # Make sure data points are matching
    sort1 = np.argsort(names1)[np.isin(np.sort(names1), names2)]
    sort2 = np.argsort(names2)[np.isin(np.sort(names2), names1)]
    names1, vals1 = names1[sort1], vals1[sort1]
    names2, vals2 = names2[sort2], vals2[sort2]
    names = names1
    
    # Define the column that is used as the data
    scol1, scol2 = -1, -1
    if '--column' in sys.argv:
        scol1 = sys.argv[sys.argv.index('--column')+1]
        scol2 = sys.argv[sys.argv.index('--column')+2]
        scol1 = get_index(scol1, header1)
        scol2 = get_index(scol2, header2)
    
    if header1 is not None:
        print('Selected column for x', header1[scol1])
    if header2 is not None:
        print('Selected column for y', header2[scol2])
    
    vals1 = vals1[:,scol1].astype(float)
    vals2 = vals2[:,scol2].astype(float)
    print('Shape data', np.shape(vals1), np.shape(vals2))

    # Read in file with defined set of data points
    if '--select_set' in sys.argv:
        sset = np.genfromtxt(sys.argv[sys.argv.index('--select_set')+1], dtype = str)
        sort = np.isin(names, sset)
        names, vals1, vals2 = names[sort], vals1[sort], vals2[sort]
  
    # Define value for inf
    posinf = 1e16
    if '--inf' in sys.argv:
        posinf = float(sys.argv[sys.argv.index('--inf')+1])
    vals1 = np.nan_to_num(vals1, posinf = posinf, neginf = -posinf)
    vals2 = np.nan_to_num(vals2, posinf = posinf, neginf = -posinf)
  
    # Decide if data should be transformed in different ways
    if '--similarity' in sys.argv:
        vals1 = 1.-vals1
        vals2 = 1.-vals2
    if '--similarityX' in sys.argv:
        vals1 = 1.-vals1
    if '--similarityY' in sys.argv:
        vals2 = 1.-vals2

    if '--xlog10' in sys.argv:
        vals1 = np.log10(vals1)
    if '--ylog10' in sys.argv:
        vals2 = np.log10(vals2)
    if '--xlog10plus' in sys.argv:
        vals1 = np.log10(vals1+1)
    if '--ylog10plus' in sys.argv:
        vals2 = np.log10(vals2+1)
    
    # If number are discrete, add some uniform noise to show all data points
    if '--wigglex' in sys.argv:
        vals1 = vals1 + 0.6* np.random.random(len(vals1))-0.3
    if '--wiggley' in sys.argv:
        vals2 = vals2 + np.random.random(len(vals2))-0.5
    if '--wigglexy' in sys.argv:
        rad = np.random.random(len(vals1)) *0.45
        radratio = 2*np.random.random(len(vals1)) - 1
        radratio2 = np.sqrt(1.-radratio**2)
        vals1 = vals1 + rad * radratio
        vals2 = vals2 + rad * radratio2
    

    vals = np.array([vals1, vals2])
    
    
    # Provide sizes to data points as additional dimension
    if '--sizefile' in sys.argv:
        sizefile = sys.argv[sys.argv.index('--sizefile')+1]
        scol = int(sys.argv[sys.argv.index('--sizefile')+2]) # column the data file
        sqrt = sys.argv[sys.argv.index('--sizefile')+3] == 'True' # determine sizes should be scaled by square root
        snames, header, sizes = read_matrix_file(sizefile)
        sizes = sizes[:,scol]
        sort = np.argsort(snames)[np.isin(np.sort(snames), names)] # sort to original data
        snames, sizes = snames[sort], sizes[sort]
        if not np.isin(snames, names).all():
            print('Sizes are incomplete. Only present', int(np.sum(np.isin(snames, names))))
            sys.exit()
        if sqrt:
            sizes = np.sqrt(sizes) + 1
        if '--adjust_size' in sys.argv:
            sizes = sizes * float(sys.argv[sys.argv.index('--adjust_size')+1])
    
    elif '--size' in sys.argv: # adjust size of all data points 
        sizes = np.ones(len(vals[0]))*float(sys.argv[sys.argv.index('--size')+1])
    else:
        sizes = None
        
    
    
    # Add color as additional dimension to scatter plot
    if '--colorfile' in sys.argv: # Loads file in which a defined column defines the color
        colorfile = sys.argv[sys.argv.index('--colorfile')+1]
        ccol = sys.argv[sys.argv.index('--colorfile')+2]
        cdelimiter = None
        if '--colordelimiter' in sys.argv: # Set delimiter of file if needed
            cdelimiter = sys.argv[sys.argv.index('--colordelimiter')+1]
        cnames, header, colors = read_matrix_file(colorfile, delimiter = cdelimiter)
        
        # Automatically find the right column in the header
        if ccol == 'find' and header1 is not None:
            ccol = find_in_list(header1[scol1], header)
            if ccol is None:
                ccol = find_in_list(header2[scol2], header)
        else:
            ccol = int(ccol)
            
        if header is not None:
            print('Color column', header[ccol])
        colors = colors[:,ccol]
        sort = np.argsort(cnames)[np.isin(cnames, names)]
        cnames, colors = cnames[sort], colors[sort]
        
        if not np.array_equal(cnames, names):
            print('Colors are incomplete. Only present', len(cnames))
            for name in names[~np.isin(names, cnames)]:
                print(name)
            sys.exit()
        if '--abscolors' in sys.argv:
            colors = np.absolute(colors) # absolute value of the values
        if '--setnancolorto' in sys.argv: # set nans if in data
            colors = np.nan_to_num(colors, float(sys.argv[sys.argv.index('--setnancolorto')+1]))
        maxcolor = round(np.amax(colors), 2)
        mincolor = round(np.amin(colors), 2)
        # if center colors, max and min will be defined by abs(max,min)
        if '--center_colors' in sys.argv and mincolor < 0:
            maxmax = max(np.absolute(mincolor), maxcolor)
            mincolor, maxcolor = -maxmax, maxmax
        minmaxcolor = round((maxcolor + mincolor)/2, 2)
        vmin, vmax = mincolor, maxcolor
        
    elif '--colorlist' in sys.argv: # Loads a list of data point names that will be assigned a 1, while others stay 0.
        colorfile = np.genfromtxt(sys.argv[sys.argv.index('--colorlist')+1], dtype = str)
        if len(np.shape(colorfile)) > 2:
            colorfile = colorfile[:,0]
        colors = np.zeros(len(names))
        colors[np.isin(names, colorfile)] = 1
        vmin, vmax = 0, 1
        
    elif '--density' in sys.argv or '--logdensity' in sys.argv: # Will color data points by density 
        colors = gaussian_kde(vals[:, np.random.permutation(len(vals[0]))[:3000]])(vals)
        if '--logdensity' in sys.argv:
            colors = np.log(1+colors)
        maxcolor = round(np.amax(colors), 2)
        mincolor = round(np.amin(colors), 2)
        minmaxcolor = round((maxcolor + mincolor)/2, 2)
        colors -= np.amin(colors)
        colors /= np.amax(colors)
        vmin, vmax = 0, 1
    
    elif '--logdensity' in sys.argv:
        colors = np.log(1+gaussian_kde(vals[:, np.random.permutation(len(vals[0]))[:3000]])(vals))
        maxcolor = round(np.amax(colors), 2)
        mincolor = round(np.amin(colors), 2)
        minmaxcolor = round((maxcolor + mincolor)/2, 2)
        colors -= np.amin(colors)
        colors /= np.amax(colors)
        vmin, vmax = 0, 1
    
    elif '--setcolor' in sys.argv:
        cmap = ListedColormap([sys.argv[sys.argv.index('--setcolor')+1]])
        colors = np.ones(len(vals[0]))
        vmin = vmax = None
    else:
        colors = None
        vmin = vmax = None
    
    
    # Specify figure and subplot properties
    s1, s2 = 4.5,4.5
    if '--figsize' in sys.argv:
        s1, s2 = np.array(sys.argv[sys.argv.index('--figsize')+1].split(','), dtype = float)
    
    fig = plt.figure(figsize = (s1,s2), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    alpha = 0.7
    if '--alpha' in sys.argv:
        alpha = float(sys.argv[sys.argv.index('--alpha')+1])
    
    lw = 0.5
    if '--lw' in sys.argv:
        lw = float(sys.argv[sys.argv.index('--lw')+1])
        
    cmap = 'viridis'
    if '--cmap' in sys.argv:
        cmap = sys.argv[sys.argv.index('--cmap')+1]
        if ',' in cmap:
            cmap = ListedColormap(cmap.split(','))
    
    if '--classcolormap' in sys.argv: # Transform colormap into specific classes 
        ncol = len(np.unique(colors))
        cmap = cm.get_cmap(cmap, ncol)
        cmap = ListedColormap(cmap.colors)
    
    if '--vlim' in sys.argv:
        vlim = sys.argv[sys.argv.index('--vlim')+1].split(',')
        vmin, vmax = float(vlim[0]), float(vlim[1])
        mincolor, maxcolor, minmaxcolor = vmin, vmax, (vmin+vmax)/2
    
    # Sort the data points based on their color strength
    sortd = np.arange(len(vals[0]), dtype = int)
    if colors is not None:
        if len(colors) == len(vals[0]):
            if '--sortascending' in sys.argv:
                sortd = np.argsort(-np.absolute(colors))
            if '--sortdescending' in sys.argv:
                sortd = np.argsort(-np.absolute(colors))
            else:
                sortd = np.argsort(np.absolute(colors))
        
    
    if colors is not None:
        colors = colors[sortd]
    if sizes is not None:
        sizes = sizes[sortd]
    
    a = ax.scatter(vals[0][sortd], vals[1][sortd], s = sizes, cmap = cmap, c=colors, alpha = alpha, vmin = vmin, vmax = vmax, edgecolor = 'silver', lw = lw, label = 'R:'+str(round(pearsonr(vals[0], vals[1])[0],2)))
    
    # Add a boxplot with bins along x to the scatter plot
    if '--boxplot' in sys.argv:
        nbins = int(sys.argv[sys.argv.index('--boxplot')+1])
        xlim = sys.argv[sys.argv.index('--boxplot')+2].split(',') # Define the boundaries as start,end
        windows = np.linspace(float(xlim[0]), float(xlim[1]), nbins +1)
        wticks = (windows[1:]+windows[:-1])/2
        boxes = [vals[1, (vals[0]>=windows[n]) * (vals[0]<=windows[n+1])] for n in range(nbins)]
        ab = ax.boxplot(boxes, positions = wticks, zorder = 1)
        ax.set_xticks(wticks) # if Boxplot is added, xticks become the center of boxplot
        ax.set_xticklabels(np.around(wticks,5).astype(str),rotation = 90)

    # Add a density contour on top of the scatter plot
    elif '--contour' in sys.argv:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ('--density' in sys.argv) or ('--logdensity' in sys.argv):
            density = colors
        else:
            if 'log' in sys.argv[sys.argv.index('--contour')+1]:
                density = np.log(1+gaussian_kde(vals[:, np.random.permutation(len(vals[0]))[:3000]])(vals))
            else:
                density = gaussian_kde(vals[:, np.random.permutation(len(vals[0]))[:3000]])(vals)
            density = density[sortd]
        ax.tricontour(vals[0][sortd], vals[1][sortd], density, levels=14, linewidths=0.5, colors='k', zorder = 1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    elif '--splitcontour' in sys.argv and '--colorfile' in sys.argv: # This adds two contour maps on top
        # and splits them into points with positive and negative color
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if '--weight_splitcontour' in sys.argv: # weights data points by their color intensity
            densweight = np.copy(colors)
            densweightvar = np.mean((densweight-minmaxcolor)**2)
            weightmask = np.absolute(densweight)<1.3*np.sqrt(densweightvar)
            densweight = 0. + np.absolute((densweight - minmaxcolor)**2/densweightvar)
            densweight[weightmask] = 0
        else:
            densweight = np.ones(len(vals[0]))
        posmask = np.random.permutation(np.where(colors >= minmaxcolor)[0])[:3000]
        posdensity = gaussian_kde(vals[:, posmask], weights = densweight[posmask])(vals)
        negmask = np.random.permutation(np.where(colors <= minmaxcolor)[0])[:3000]
        negdensity = gaussian_kde(vals[:, negmask], weights = densweight[negmask])(vals)
        if 'log' in sys.argv[sys.argv.index('--splitcontour')+1]:
            posdensity = np.log(1+posdensity)
            negmask = np.random.permutation(np.where(colors <= minmaxcolor)[0])[:3000]
            negdensity = np.log(1+negdensity)
            
        posdensity = posdensity[sortd]
        negdensity = negdensity[sortd]
        ax.tricontour(vals[0][sortd], vals[1][sortd], posdensity, levels=6, linewidths=0.5, colors = [cm.get_cmap(cmap)(1.)])
        ax.tricontour(vals[0][sortd], vals[1][sortd], negdensity, levels=6, linewidths=0.5, colors = [cm.get_cmap(cmap)(0.)])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    
    
    ax.set_xlabel(namex)
    ax.set_ylabel(namey)
    
    if '--legend' in sys.argv:
        ax.legend()
    
    
    if '--colorbar' in sys.argv and colors is not None: # Add colorbar manually if colorfile was given
        colorbartitle = sys.argv[sys.argv.index('--colorbar')+1]
        pax = ax.get_position()
        ax2 = fig.add_subplot(555)
        ax2.set_position([pax.x0+pax.width, pax.y0+pax.height*5/6, pax.width * 1/12., pax.height*1/6.])
        ax2.imshow(np.linspace(0,1,100)[::-1].reshape(-1,1), cmap = cmap, vmin = 0, vmax = 1, aspect= 'auto')
        ax2.set_title(colorbartitle)
        ax2.set_yticks([0,50,100])
        ax2.set_yticklabels([maxcolor,minmaxcolor,mincolor])
        ax2.tick_params(left = False, labelleft = False, right = True, labelright = True, bottom = False, labelbottom = False)
    
    if '--xlim' in sys.argv:
        xlim = sys.argv[sys.argv.index('--xlim')+1].split(',')
        xlim = np.array(xlim, dtype = float)
        ax.set_xlim(xlim)
        
    if '--ylim' in sys.argv:
        ylim = sys.argv[sys.argv.index('--ylim')+1].split(',')
        ylim = np.array(ylim, dtype = float)
        ax.set_ylim(ylim)
    
    if '--xysamelim' in sys.argv:
        xlim = [np.amin(vals), np.amax(vals)]
        ylim = [np.amin(vals), np.amax(vals)]
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    # Add axis and diagonals as orientation
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if '--zeroxaxis' in sys.argv:
        ax.plot(xlim, [0,0], color = 'grey')
    if '--zeroyaxis' in sys.argv:
        ax.plot([0,0], ylim, color = 'grey')
        
    lim = [np.amax(np.amin(vals,axis = 1)), np.amin(np.amax(vals,axis = 1))]
    if '--plotdiagonal' in sys.argv:
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color = 'maroon')
    if '--plotnegdiagonal' in sys.argv:
        ax.plot([lim[0], lim[1]], [lim[1], lim[0]], color = 'maroon')    
    
    # Change scale of axis
    if '--logx' in sys.argv:
        ax.set_xscale('log')
    if '--symlogx' in sys.argv:
        ax.set_xscale('symlog')
    if '--logy' in sys.argv:
        ax.set_yscale('log')
    if '--symlogy' in sys.argv:
        ax.set_yscale('symlog')
    
    # Add the names of single data point as text next to them in the plot
    if '--addnamestoscatter' in sys.argv:
        # try this to get non-ovelapping: https://github.com/Phlya/adjustText
        topnames = None
        if len(sys.argv) >sys.argv.index('--addnamestoscatter')+1:
            if isint(sys.argv[sys.argv.index('--addnamestoscatter')+1]):
                topnames = int(sys.argv[sys.argv.index('--addnamestoscatter')+1])
            if len(sys.argv) >sys.argv.index('--addnamestoscatter')+2:
                sorttopnames = sys.argv[sys.argv.index('--addnamestoscatter')+2]
                if sorttopnames.upper() == 'X':
                    mask = np.sort(-np.absolute(vals[0], axis = 0))[:topnames]
                if sorttopnames.upper() == 'Y':
                    mask = np.sort(-np.absolute(vals[1], axis = 0))[:topnames]
                if sorttopnames == 'color':
                    mask = sortd[::-1][:topnames]
            else:
                mask = np.sort(-np.absolute(np.sum(vals**2, axis = 0)))[:topnames]
        if mask is None:
            mask = np.arange(names)
        for n in mask:
            ax.text(vals[0][n], vals[1][n], names[n], va='bottom', ha = 'left')
    
    # Specify figure output
    dpi = 300
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
    
    fmt = 'jpg'
    if '--format' in sys.argv:
        fmt = sys.argv[sys.argv.index('--format')+1]
    
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        print(outname+'_scatter.'+fmt)
        fig.savefig(outname+'_scatter.'+fmt, dpi = dpi, bbox_inches = 'tight')
    else:
        #plt.tight_layout()
        plt.show()
        
