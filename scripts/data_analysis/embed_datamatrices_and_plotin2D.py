'''
Create embedding and data matrix and plot 2D scatter plot with umap
TODO 
Use drg_tools.plotlib scatterPlot

'''

import numpy as np
import sys, os
import matplotlib.pyplot as plt

from drg_tools.data_processing import reduce_dim, embedding
from drg_tools.io_utils import check, readalign_matrix_files

from functools import reduce


    

def plot_2d(orig_2d, colors = None, cmap = None, alpha = 0.5, figsize = (3.5,3.5), size = 5, outname = None, dpi = 200, vlim = None, sortbycolor = 0):
    '''
    Simple 2D scatter plot for visualization
    '''
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if vlim is None:
        if colors is not None:
            vmin, vmax = np.amin(colors), np.amax(colors)
            if vmin < 0:
                vabs = np.amax(abs(vmin), abs(vmax))
                vmin, vmax = -vabs, vabs
            print(vmin, vmax)
    if sortbycolor != 0:
        if isinstance(colors, np.ndarray):
            sortcolors = np.copy(colors)
            if abs(sortbycolor) > 1:
                sortcolors = np.absolute(colors)
            if sortbycolor < 0:
                sortcolors = -sortcolors
            sort = np.argsort(sortcolors)
            colors, orig_2d = colors[sort], orig_2d[sort]
            
    ap = ax.scatter(orig_2d[:,0], orig_2d[:,1], c = colors, alpha = alpha, cmap = cmap, s = size)
    fig.colorbar(ap, aspect = 2, pad = 0, anchor = (0,0.9), shrink = 0.15)
    if outname is not None:
        fig.savefig(outname+'_2d.jpg', dpi = dpi)
    else:
        plt.show()
    
    




if __name__ == '__main__':
    
    names, colnames, x_emb = readalign_matrix_files(sys.argv[1])
    
    outname = os.path.splitext(sys.argv[1])[0]
    if '--embedded' in sys.argv:
        xy = x_emb
    else:
        params = {"norm" : 'original', "normvar" : 1., "algorithm" : 'umap', "n_components" : 2}
        if '--embedding' in sys.argv:
            embmethparms = sys.argv[sys.argv.index('--embedding')+1]
            if '+' in embmethparms:
                embmethparms=embmethparms.split('+')
            else:
                embmethparms=[embmethparms]

            for e, embm in enumerate(embmethparms):
                if '=' in embm:
                    embm = embm.split('=')
                elif ':' in embm:
                    embm = embm.split(':')
                params[embm[0]] = check(embm[1])
                print(embm[0], check(embm[1]))
        for p, par in enumerate(params):
            outname += '_'+str(par)[:2] + str(params[par])[:3]
        
        print(params)
        embed = embedding(**params)
        xy = embed.fit(x_emb)
    
        if '--save_embedding' in sys.argv:
            np.savez_compressed(outname, names = names, values = xy)
    
    if '--addname' in sys.argv:
        outname += '_'+sys.argv[sys.argv.index('--addname') + 1]
    print(outname)
    
    #include option for various colormaps that can be mixed in one umap.
    viskwargs = {}
    if '--colors' in sys.argv:
        cfile = sys.argv[sys.argv.index('--colors')+1]
        ccont = int(sys.argv[sys.argv.index('--colors')+2])
        outname += 'col'+str(ccont)
        if os.path.splitext(cfile)[-1] == '.npz':
            cnames, colorcnames, colors = readalign_matrix_files(cfile)
            colors = colors[:, ccont]
            print(cnames, colors)
        else:
            cfile = np.genfromtxt(cfile, dtype = str)
            cnames, colors = cfile[:,0], cfile[:, ccont].astype(float)
        sort = np.argsort(cnames)[np.isin(np.sort(cnames),names)]
        cnames, colors = cnames[sort], colors[sort]
        if not np.array_equal(cnames,names):
            print('Colors dont match names in embedding')
            sys.exit()
        else:
            viskwargs['colors'] = colors
        if '--transform_color' in sys.argv:
            colors = 1. - colors
    
    if '--cmap' in sys.argv:
        cmap = sys.argv[sys.argv.index('--cmap')+1]
        if ',' in cmap:
            cmap = ListedColormap(cmap.split(','))
        viskwargs['cmap'] = cmap
    
    if '--plot_params' in sys.argv:
        visparms = sys.argv[sys.argv.index('--plot_params')+1]
        if '+' in visparms:
            visparms=visparms.split('+')
        else:
            visparms=[visparms]
        for v, vm in enumerate(visparms):
            if '=' in vm:
                vm = vm.split('=')
            elif ':' in embm:
                vm = vm.split(':')
            viskwargs[vm[0]] = check(vm[1])
    for p, par in enumerate(viskwargs):
        if not isinstance(viskwargs[par], np.ndarray) and not isinstance(viskwargs[par], list):
            outname += '_'+str(par)[:2] + str(viskwargs[par])[:3]
    
    if '--savefig' in sys.argv:
        viskwargs['outname'] = outname
        if '--outname' in sys.argv:
            viskwargs['outname'] = sys.argv[sys.argv.index('outname')+1]
            
    plot_2d(xy, **viskwargs)

    
        
