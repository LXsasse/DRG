import numpy as np
import matplotlib.pyplot as plt 
import sys, os
from matrix_plot import plot_heatmap
from data_processing import check 
from functools import reduce 

def read(outputfile, delimiter = None):
    celltypes = None
    if os.path.isfile(outputfile):
        if os.path.splitext(outputfile)[1] == '.npz':
            Yin = np.load(outputfile, allow_pickle = True)
            if 'values' in Yin.files:
                Y = Yin['values']
            else:
                Y = Yin['counts']
            outputnames = Yin['names'] # Y should of shape (nexamples, nclasses, l_seq/n_resolution)
            if 'celltypes' in Yin.files:
                celltypes = Yin['celltypes']
            elif 'columns' in Yin.files:
                celltypes = Yin['columns']
        else:
            Yin = np.genfromtxt(outputfile, dtype = str, delimiter = delimiter)
            Y, outputnames = Yin[:, 1:].astype(float), Yin[:,0]
            Yf = open(outputfile).readline()
            if Yf[0] == '#':
                celltypes = np.array(Yf.strip('#').strip().split())
    else:
        if ',' in outputfile:
            Y, outputnames, celltypes = [], [], []
            for putfile in outputfile.split(','):
                if os.path.splitext(putfile)[1] == '.npz':
                    Yin = np.load(putfile, allow_pickle = True)
                    onames = Yin['names']
                    sort = np.argsort(onames)
                    Y.append(Yin['counts'][sort])
                    outputnames.append(onames[sort])
                    if 'celltypes' in Yin.files:
                        celltypes.append(Yin['celltypes'])
                else:
                    Yin = np.genfromtxt(putfile, dtype = str, delimiter = delimiter)
                    onames = Yin[:,0]
                    sort = np.argsort(onames)
                    Y.append(Yin[:, 1:].astype(float)[sort]) 
                    outputnames.append(onames[sort])
                    Yf = open(outputfile).readline()
                    if Yf[0] == '#':
                        celltypes.append(np.array(Yf.strip('#').strip().split()))
                
            comnames = reduce(np.intersect1d, outputnames)
            for i, yi in enumerate(Y):
                Y[i] = yi[np.isin(outputnames[i], comnames)]
            outputnames = comnames
            if len(celltypes) == 0:
                celltypes = None
            else:
                celltypes = np.concatenate(celltypes)
    Y = np.concatenate(Y, axis = 1)
    print(len(outputnames), np.shape(Y))
    return outputnames, Y, celltypes

def sortafter(given, target):
    sort = []
    given = list(given)
    for t, tar in enumerate(target):
        sort.append(given.index(tar))
    return np.array(sort)

if __name__ == '__main__': 
    
    data = sys.argv[1]
    
    ynames, X, xnames = read(data)
    
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
        ydnames, ydistmat, xdnames = read(sys.argv[sys.argv.index('--ydistmat')+1])
        ysort, xsort = sortafter(ydnames, ynames), sortafter(xdnames, xnames)
        ydistmat = ydistmat[ysort][:, xsort]
        cparms['ydistmat'] = ydistmat
    
    if '--xdistmat' in sys.argv:
        dnames, xdistmat, xdnames = read(sys.argv[sys.argv.index('--xdistmat')+1])
        ysort, xsort = sortafter(ydnames, ynames), sortafter(xdnames, xnames)
        xdistmat = xdistmat[ysort][:, xsort]
        cparms['xdistmat'] = xdistmat
        
    
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
    
    
    
