import numpy as np
import matplotlib.pyplot as plt 
import sys, os
from drg_tools.plotlib import plot_heatmap
from drg_tools.io_utils import check, readalign_matrix_files, sortafter
from functools import reduce 





if __name__ == '__main__': 
    
    data = sys.argv[1]
    
    ynames, xnames, X = readalign_matrix_files(data)
    
    if '--list' in sys.argv:
        tlist = np.genfromtxt(sys.argv[sys.argv.index('--list')+1], dtype = str)
        mask = np.isin(ynames, tlist)
        ynames, X = ynames[mask], X[mask]
    
    cparms = {}
    if '--matrix_features' in sys.argv:
        matparms = sys.argv[sys.argv.index('--matrix_features')+1]
        if '+' in matparms:
            matparms = matparms.split('+')
        else:
            matparms = [matparms]
        
        for c, clp in enumerate(matparms):
            clp = clp.split('=')
            if clp[0] == 'yticklabels':
                if clp[1] == 'True' and ynames is not None:
                    cparms[clp[0]] = ynames
            elif clp[0] == 'xticklabels':
                if clp[1] == 'True' and ynames is not None:
                    cparms[clp[0]] = xnames
            else:
                cparms[clp[0]] = check(clp[1])
            
    if '--ydistmat' in sys.argv:
        ydnames, xdnames, ydistmat = readalign_matrix_files(sys.argv[sys.argv.index('--ydistmat')+1])
        ysort = sortafter(ydnames, ynames)
        ydistmat = ydistmat[ysort]
        cparms['ydistmat'] = ydistmat
    
    if '--xdistmat' in sys.argv:
        dnames, xdnames, xdistmat = readalign_matrix_files(sys.argv[sys.argv.index('--xdistmat')+1])
        xsort = sortafter(xdnames, xnames)
        xdistmat = xdistmat[:, xsort]
        cparms['xdistmat'] = xdistmat
    
    x_attributes = None
    if '--xattributes' in sys.argv:
        xattributes = sys.argv[sys.argv.index('--xattributes')+1]
        if os.path.isfile(xattributes):
            xattributes = np.genfromtxt(xattributes, dtype = str)
        elif xattributes == 'names':
            x_attributes = np.array([[xn.rsplit('.')[0].upper(),xn.rsplit('.')[-1].upper()] for xn in xnames]).T
        keep = []
        for t, xat in enumerate(x_attributes):
            print(np.unique(xat))
            if len(np.unique(xat)) > 1:
                keep.append(t)
        x_attributes = x_attributes[keep]
        cparms['x_attributes'] = x_attributes\

    y_attributes = None
    if '--yattributes' in sys.argv:
        yattributes = sys.argv[sys.argv.index('--yattributes')+1]
        if os.path.isfile(yattributes):
            y_attributes = np.genfromtxt(yattributes, dtype = str)
            mask = np.isin(y_attributes[:,0], ynames)
            y_attributes = y_attributes[mask]
            if not np.array_equal(y_attributes[:,0], ynames):
                keep = []
                for jy, yn in enumerate(ynames):
                    keep.append(list(y_attributes[:,0]).index(yn))
                y_attributes = y_attributes[keep]
            y_attributes = y_attributes[:,[1]]
            print(np.unique(y_attributes))
            cparms['y_attributes'] = y_attributes
    
    
    figname = None
    if '--savefig' in sys.argv:
        figname = sys.argv[sys.argv.index('--savefig')+1]
        
    plot_heatmap(X,# matrix that is plotted with imshow
                 figname = figname, 
                 **cparms)
'''
                 ydistmat = None,
                 xdistmat = None,
                 measurex = None, # if matrix is not a symmetric distance matrix then measurex define distannce metric for linkage clustering 
                 measurey = None, # same as measurex just for y axic
                 sortx = None, # agglomerative clustering algorith used in likage, f.e average, or single
                 sorty = None, # same as above but for y axis
                 x_attributes = None, # additional heatmap with attributes of columns
                 y_attributes = None, # same as above for y axis
                 xattr_name = None, # names of attributes for columns
                 yattr_name = None, # names of attributes for rows
                 heatmapcolor = 'BrBG_r', # color map of main matrix
                 xatt_color = None, # color map or list of colormaps for attributes
                 yatt_color = None, 
                 xatt_vlim = None, # vmin and vmas for xattributes, or list of vmin and vmax
                 yatt_vlim = None,
                 pwms = None, # pwms that are plotted with logomaker next to rows of matrix
                 infocont = True, # if True, the matrices will be plotted as information content
                 combine_cutx = 0., # NOT implemented yet, can be used to cut off linkage tree at certain distance if linkage tree too full
                 combine_cuty = 0., 
                 color_cutx = 0., # cut off for coloring in linkage tree. 
                 color_cuty = 0., 
                 xdenline = None, # line drawn into linkage tree on x-axis
                 ydenline = None, 
                 plot_value = False, # if true the values are written into the cells of the matrix
                 vmin = None, # min color value 
                 vmax = None, 
                 grid = False, # if True, grey grid drawn around heatmap cells
                 xlabel = None, # label on x-axis
                 ylabel = None, # ylabel
                 xticklabels = None,
                 yticklabels  = None,
                 showdpi = None,
                 dpi = None,
                 figname = None,
                 fmt = '.jpg',
                 maxsize = 150, 
                 cellsize = 0.3,
                 noheatmap = False,
                 row_distributions = None,
                 row_distribution_kwargs = {})
'''
    
    
    
