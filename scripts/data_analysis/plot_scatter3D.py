# plot3d.py
import numpy as np
import sys, os
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

from functools import reduce 
from drg_tools.io_utils import read_matrix_files
from drg_tools.plotlib import plot3d

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
    
    names1, header1, vals1 = read_matrix_files(file1, subtract = subt, delimiter = delimiter)
    names2, header2, vals2 = read_matrix_files(file2, subtract = subt, delimiter = delimiter)
    names3, header3, vals3 = read_matrix_files(file3, subtract = subt, delimiter = delimiter)
    
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
        cnames, header, colors = read_matrix_files(colorfile, delimiter = cdelimiter)
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

    

    
    
