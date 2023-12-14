# plot3d.py
import numpy as np
import sys, os
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.stats import gaussian_kde, pearsonr
from functools import reduce 
from scatter_comparison_plot import readfiles



def scatter3D(x,y,z, axis = True, color = None, cmap = None, xlabel = None, ylabel = None, zlabel=None, alpha = 0.9, diag = False):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(111, projection='3d') #plt.axes(projection='3d')
    xlim = np.array([np.amin(x), np.amax(x)])
    ylim = np.array([np.amin(y), np.amax(y)])
    zlim = np.array([np.amin(z), np.amax(z)])
    lrat = 0.5
    if axis:
        # plot axis in there
        xlim0, ylim0, zlim0 = xlim * lrat, ylim * lrat, zlim * lrat
        ax.plot3D(xlim0,[0,0],[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],ylim0,[0,0], color = 'k', lw = 1)
        ax.plot3D([0,0],[0,0],zlim0, color = 'k', lw = 1)
    if diag:
        maxlim = np.array([xlim, ylim,zlim])*lrat
        print(maxlim)
        maxlim = [np.amax(maxlim[:,0]), np.amin(maxlim[:,1])]
        print(maxlim)
        ax.plot3D(maxlim, maxlim, maxlim, color = 'maroon', lw = 1)
    # plot a scatterplot in 3d
    ax.scatter3D(x, y, z, c=color, cmap=cmap, lw=0, alpha = alpha, s = 3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    lrat = 0.75
    ax.set_xlim(xlim*lrat)
    ax.set_xlim(ylim*lrat)
    ax.set_xlim(zlim*lrat)
    ax.view_init(elev=25, azim=-49)
    return fig


if __name__ == '__main__':    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    namex = sys.argv[4]
    namey = sys.argv[5]
    namez = sys.argv[6]
    
    delimiter = None
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    
    subt = False
    if '--similarity' in sys.argv:
        subt = True
    
    names1, vals1, header1 = readfiles(file1, subtract = subt, delimiter = delimiter)
    names2, vals2, header2 = readfiles(file2, subtract = subt, delimiter = delimiter)
    names3, vals3, header3 = readfiles(file3, subtract = subt, delimiter = delimiter)
    
    scol1, scol2, scol3 = -1, -1, -1
    if '--column' in sys.argv:
        scol1 = int(sys.argv[sys.argv.index('--column')+1])
        scol2 = int(sys.argv[sys.argv.index('--column')+2])
        scol3 = int(sys.argv[sys.argv.index('--column')+3])
    
    if header1 is not None:
        print(header1[scol1])
    if header2 is not None:
        print(header2[scol2])
    if header3 is not None:
        print(header3[scol3])
    
    
    vals1 = vals1[:,scol1]
    vals2 = vals2[:,scol2]
    vals3 = vals3[:,scol3]
    common = reduce(np.intersect1d, [names1, names2, names3])
    sort1 = np.isin(names1, common)
    sort2 = np.isin(names2, common)
    sort3 = np.isin(names3, common)
    names, vals1, vals2, vals3 = names1[sort1], vals1[sort1], vals2[sort2], vals3[sort3]
    
    
    if '--select_set' in sys.argv:
        sset = np.genfromtxt(sys.argv[sys.argv.index('--select_set')+1], dtype = str)
        sort = np.isin(names, sset)
        names, vals1, vals2, vals3 = names[sort], vals1[sort], vals2[sort], vals3[sort]
    
        
    if '--xlog' in sys.argv:
        vals1 = np.log10(vals1)
    if '--ylog' in sys.argv:
        vals2 = np.log10(vals2)
    if '--zlog' in sys.argv:
        vals3 = np.log10(vals3)
    
    vals1 = np.nan_to_num(vals1)
    vals2 = np.nan_to_num(vals2)
    vals3 = np.nan_to_num(vals3)

    colors = None
    if '--colorfile' in sys.argv:
        colorfile = sys.argv[sys.argv.index('--colorfile')+1]
        ccol = sys.argv[sys.argv.index('--colorfile')+2]
        cdelimiter = None
        if '--colordelimiter' in sys.argv:
            cdelimiter = sys.argv[sys.argv.index('--colordelimiter')+1]
        cnames, colors, header = readfiles(colorfile, delimiter = cdelimiter)
        ccol = int(ccol)
        if header is not None:
            print('Color column', header[ccol])
        colors = colors[:,ccol]
        if '--abscolors' in sys.argv:
            colors = np.absolute(colors)
        sort = np.argsort(cnames)[np.isin(cnames, names)]
        cnames, colors = cnames[sort], colors[sort]
        if not np.array_equal(cnames, names):
            print('Colors are incomplete. Only present', int(np.sum(np.isin(cnames, names))))
            ncolors = -np.ones(len(names))
            ncolors[np.isin(names, cnames)] = colors
            colors = ncolors
        maxcolor = round(np.amax(colors), 2)
        mincolor = round(np.amin(colors), 2)
        minmaxcolor = round((maxcolor + mincolor)/2, 2)
        colors -= np.amin(colors)
        colors /= np.amax(colors)
    
    cmap = 'Dark2_r'
    if '--cmap' in sys.argv:
        cmap = sys.argv[sys.argv.index('--cmap')+1]
    
    alpha = 0.9
    if '--alpha' in sys.argv:
        alpha = float(sys.argv[sys.argv.index('--alpha')+1])
        
    diag = False
    if '--plotdiagonal' in sys.argv:
        diag = True

    fig = scatter3D(vals1,vals2,vals3, axis = True, color = colors, cmap = cmap, xlabel = namex, ylabel = namey, zlabel = namez, alpha = alpha, diag = diag)
    
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        fig.savefig(outname, dpi = 300, bbox_inches = 'tight')
    
    plt.show()

    

    
    
