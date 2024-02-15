#scatter_pv_fc.py
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde, pearsonr
from matplotlib.colors import ListedColormap

def readfiles(file1, delimiter = None, subtract = False):
    fi = open(file1, 'r').readlines()
    if delimiter is None:
        if os.path.splitext(file1)[-1] == '.txt' or os.path.splitext(file1)[-1] == '.dat':
            delimiter = ' '
        elif os.path.splitext(file1)[-1] == '.tsv' or os.path.splitext(file1)[-1] == '.tab':
            delimiter = '\t'
        elif os.path.splitext(file1)[-1] == '.csv':
            delimiter = ','
        
    names1, vals1, header = [], [], None
    for l, line in enumerate(fi):
        if line[0] != '#':
            line = line.strip().split(delimiter)
            names1.append(line[0].replace('.FC.', '.').replace('.PV.', '.'))
            va = line[1:]
            if 'NA' in va:
                ava = []
                for v in va:
                    if v == 'NA':
                        ava.append('nan')
                    else:
                        ava.append(v)
                va = ava
                
            vals1.append(va)
        elif l == 0:
            header = np.array(line.strip('#').strip().split(delimiter))
    
    if header is None and ((len(vals1[0]) != len(vals1[1])) or ( not checknum(vals1[0][0]))):
        header = [names1[0]] + vals1[0]
        vals1 = vals1[1:]
        names1 = names1[1:]
    vals1 = np.array(vals1)
    if subtract:
        vals1 = 1-vals1
    names1 = np.array(names1)
    names1, sort = np.unique(names1, return_index = True)
    vals1 = vals1[sort]
    if header is not None and len(header) == np.shape(vals1)[1]:
        header, sort = np.unique(header, return_index = True)
        sort = np.argsort(sort)
        header = header[sort]
        vals1 = vals1[:, sort]
    return names1, vals1, header 

def checkint(i):
    try:
        int(i)
    except:
        return False
    return True

def checknum(i):
    try:
        int(i)
    except:
        try: 
            float(i)
        except:
            return False
    return True

def find_in_list(target, thelist):
    for l, lo in enumerate(thelist):
        if target in lo or lo in target:
            return l


if __name__ == '__main__':    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    namex = sys.argv[3]
    namey = sys.argv[4]
    
    delimiter1 = delimiter2 = None
    if '--delimiter' in sys.argv:
        delimiter1 = delimiter2 = sys.argv[sys.argv.index('--delimiter')+1]
    if '--delimiter1' in sys.argv:
        delimiter1 = sys.argv[sys.argv.index('--delimiter1')+1]
    if '--delimiter2' in sys.argv:
        delimiter2 = sys.argv[sys.argv.index('--delimiter2')+1]
    
    subtA = False
    subtB = False
    if '--similarity' in sys.argv:
        subtA = subtB = True
    if '--similarityA' in sys.argv:
        subtA = True
    if '--similarityB' in sys.argv:
        subtB = True
    
        
        
    names1, vals1, header1 = readfiles(file1, subtract = subtA, delimiter = delimiter1)
    names2, vals2, header2 = readfiles(file2, subtract = subtB, delimiter = delimiter2)
    print(names1, names2)
    
    #print(names1, names2)
    #print(np.shape(vals1), np.shape(vals2))
    #print(header1, header2)
    #sys.exit()
    # Sorting is further down
    #sort1 = np.isin(names1,names2)
    #names1, vals1 = names1[sort1], vals1[sort1]
    #sort2 = np.isin(names2,names1)
    #names2, vals2 = names2[sort2], vals2[sort2]
    
    
    scol1, scol2 = -1, -1
    if '--column' in sys.argv:
        scol1 = sys.argv[sys.argv.index('--column')+1]
        scol2 = sys.argv[sys.argv.index('--column')+2]
        if checkint(scol1):
            scol1 = int(scol1)
        elif header1 is not None:
            try:
                scol1 = list(header1).index(scol1)
            except:
                print(scol1, 'not a valid column')
                sys.exit()
        if checkint(scol2):
            scol2 = int(scol2)
        elif header2 is not None:
            try:
                scol2 = list(header2).index(scol2)
            except:
                print(scol2, 'not a valid column')
                sys.exit()
        print(scol1, scol2)
    
    if header1 is not None:
        print(header1[scol1])
    if header2 is not None:
        print(header2[scol2])
    
    header = None
    if '--columns_and_row' in sys.argv:
        scol1 = sys.argv[sys.argv.index('--columns_and_row')+1].split(',')
        scol2 = sys.argv[sys.argv.index('--columns_and_row')+2].split(',')
        if checkint(scol1[0]):
            scol1 = np.array(scol1, dtype = int)
        else:
            scol1 = np.where(np.isin(header1, scol1))[0]
        if checkint(scol2[0]):
            scol2 = np.array(scol2, dtype = int)
        else:
            scol2 = np.where(np.isin(header2, scol2))[0]
        
        srow1 = sys.argv[sys.argv.index('--columns_and_row')+3]
        srow2 = sys.argv[sys.argv.index('--columns_and_row')+4]
        if checkint(srow1):
            srow1 = int(srow1)
        else:
            srow1 = list(names1).index(srow1)
        
        names1 = names1[srow1]
        if checkint(srow2):
            srow2 = int(srow2)
        else:
            srow2 = list(names2).index(srow2)
        names2 = names2[srow2]
        vals1 = vals1[srow1, scol1].astype(float)
        vals2 = vals2[srow2, scol2].astype(float)
        header = header2[scol2]
    else:
        vals1 = vals1[:,scol1].astype(float)
        vals2 = vals2[:,scol2].astype(float)
        print(np.shape(vals1), np.shape(vals2))
        sort1 = np.isin(names1, names2)
        sort2 = np.isin(names2, names1)
        names, vals1, vals2 = names1[sort1], vals1[sort1], vals2[sort2]
        if '--select_set' in sys.argv:
            sset = np.genfromtxt(sys.argv[sys.argv.index('--select_set')+1], dtype = str)
            sort = np.isin(names, sset)
            names, vals1, vals2 = names[sort], vals1[sort], vals2[sort]

    if '--xlog10' in sys.argv:
        vals1 = np.log10(vals1)
    if '--ylog10' in sys.argv:
        vals2 = np.log10(vals2)
    if '--xlog10plus' in sys.argv:
        vals1 = np.log10(vals1+1)
    if '--ylog10plus' in sys.argv:
        vals2 = np.log10(vals2+1)
    
    
    if '--wigglex' in sys.argv:
        vals1 = vals1 + np.random.random(len(vals1))-0.5
    if '--wiggley' in sys.argv:
        vals2 = vals2 + np.random.random(len(vals2))-0.5
    if '--wigglexy' in sys.argv:
        rad = np.random.random(len(vals1)) *0.45
        radratio = 2*np.random.random(len(vals1)) - 1
        radratio2 = np.sqrt(1.-radratio**2)
        vals1 = vals1 + rad * radratio
        vals2 = vals2 + rad * radratio2
        
    vals1 = np.nan_to_num(vals1)
    vals2 = np.nan_to_num(vals2)
    
    vals = np.array([vals1, vals2])
    print('Size after sorting', np.shape(vals))
    
    if '--sizefile' in sys.argv:
        sizefile = sys.argv[sys.argv.index('--sizefile')+1]
        scol = int(sys.argv[sys.argv.index('--sizefile')+2])
        sqrt = sys.argv[sys.argv.index('--sizefile')+3] == 'True'
        snames, sizes, header = readfiles(sizefile)
        sizes = sizes[:,scol]
        sort = np.argsort(snames)[np.isin(np.sort(snames), names)]
        snames, sizes = snames[sort], sizes[sort]
        if not np.isin(snames, names).all():
            print('Sizes are incomplete. Only present', int(np.sum(np.isin(snames, names))))
            sys.exit()
        if sqrt:
            sizes = np.sqrt(sizes) + 1
    elif '--size' in sys.argv:
        sizes = np.ones(len(vals[0]))*float(sys.argv[sys.argv.index('--size')+1])
    else:
        sizes = None
        
    if sizes is not None and '--adjust_size' in sys.argv:
        sizes = sizes * float(sys.argv[sys.argv.index('--adjust_size')+1])
        
    if '--colorfile' in sys.argv:
        colorfile = sys.argv[sys.argv.index('--colorfile')+1]
        ccol = sys.argv[sys.argv.index('--colorfile')+2]
        cdelimiter = None
        if '--colordelimiter' in sys.argv:
            cdelimiter = sys.argv[sys.argv.index('--colordelimiter')+1]
        cnames, colors, header = readfiles(colorfile, delimiter = cdelimiter)
        if ccol == 'find' and header1 is not None:
            ccol = find_in_list(header1[scol1], header)
            if ccol is None:
                ccol = find_in_list(header2[scol2], header)
        else:
            ccol = int(ccol)
        if header is not None:
            print('Color column', header[ccol])
        colors = colors[:,ccol]
        if '--abscolors' in sys.argv:
            colors = np.absolute(colors)
        sort = np.argsort(cnames)[np.isin(cnames, names)]
        cnames, colors = cnames[sort], colors[sort]
        if not np.array_equal(cnames, names):
            print('Colors are incomplete. Only present', len(cnames))
            for name in names[~np.isin(names, cnames)]:
                print(name)
            sys.exit()
        if '--setnancolorto' in sys.argv:
            colors = np.nan_to_num(colors, float(sys.argv[sys.argv.index('--setnancolorto')+1]))
        maxcolor = round(np.amax(colors), 2)
        mincolor = round(np.amin(colors), 2)
        if '--center_colors' in sys.argv and mincolor < 0:
            maxmax = max(np.absolute(mincolor), maxcolor)
            mincolor, maxcolor = -maxmax, maxmax
        minmaxcolor = round((maxcolor + mincolor)/2, 2)
        vmin = mincolor
        vmax = maxcolor
    elif '--colorlist' in sys.argv:
        colorfile = np.genfromtxt(sys.argv[sys.argv.index('--colorlist')+1], dtype = str)
        if len(np.shape(colorfile)) > 2:
            colorfile = colorfile[:,0]
        colors = np.zeros(len(names))
        colors[np.isin(names, colorfile)] = 1
        
    elif '--density' in sys.argv:
        colors = gaussian_kde(vals[:, np.random.permutation(len(vals[0]))[:3000]])(vals)
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
    else:
        colors = None
        vmin = vmax = None
    
    
    fig = plt.figure(figsize = (4.5,4.5), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    alpha = 0.25
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
    
    if '--classcolors' in sys.argv:
        ncol = len(np.unique(colors))
        cmap = cm.get_cmap(cmap, ncol)
        cmap = ListedColormap(cmap.colors)
      
    if '--vlim' in sys.argv:
        vlim = sys.argv[sys.argv.index('--vlim')+1].split(',')
        vmin, vmax = float(vlim[0]), float(vlim[1])
        mincolor, maxcolor, minmaxcolor = vmin, vmax, (vmin+vmax)/2
    
    
    sortd = np.arange(len(vals[0]), dtype = int)
    if colors is not None:
        if len(colors) == len(vals[0]):
            if '--sortreverse' in sys.argv:
                sortd = np.argsort(-np.absolute(colors))    
            else:
                sortd = np.argsort(np.absolute(colors))
        
    
    if colors is not None:
        colors = colors[sortd]
        print(colors)
    if sizes is not None:
        sizes = sizes[sortd]
    
    
    if '--boxplot' in sys.argv:
        nbins = int(sys.argv[sys.argv.index('--boxplot')+1])
        xlim = sys.argv[sys.argv.index('--boxplot')+2].split(',')
        windows = np.linspace(float(xlim[0]), float(xlim[1]), nbins +1)
        wticks = (windows[1:]+windows[:-1])/2
        boxes = [vals[1, (vals[0]>=windows[n]) * (vals[0]<=windows[n+1])] for n in range(nbins)]
        ab = ax.boxplot(boxes, positions = wticks) #, label = 'R:'+str(round(pearsonr(vals[0], vals[1])[0],2)))
        ax.set_xticks(wticks)
        ax.set_xticklabels(np.around(wticks,5).astype(str),rotation = 90)
        a = ax.scatter(vals[0][sortd], vals[1][sortd], s = sizes, cmap = cmap, c=colors, alpha = 0.05, vmin = vmin, vmax = vmax, edgecolor = 'silver', lw = lw, label = 'R:'+str(round(pearsonr(vals[0], vals[1])[0],2)), zorder = -1)
    else:
        a = ax.scatter(vals[0][sortd], vals[1][sortd], s = sizes, cmap = cmap, c=colors, alpha = alpha, vmin = vmin, vmax = vmax, edgecolor = 'silver', lw = lw, label = 'R:'+str(round(pearsonr(vals[0], vals[1])[0],2)))
    
    if '--contour' in sys.argv:
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
        ax.tricontour(vals[0][sortd], vals[1][sortd], density, levels=14, linewidths=0.5, colors='k')
        #ax.tricontourf(vals[0][sortd], vals[1][sortd], colors, levels=14, cmap=cmap, alpha = 0.2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    if '--splitcontour' in sys.argv and '--colorfile' in sys.argv:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        densweight = np.copy(colors)
        densweightvar = np.mean((densweight-minmaxcolor)**2)
        weightmask = np.absolute(densweight)<1.3*np.sqrt(densweightvar)
        densweight = 0. + np.absolute((densweight - minmaxcolor)**2/densweightvar)
        densweight[weightmask] = 0
        if 'log' in sys.argv[sys.argv.index('--splitcontour')+1]:
            posmask = np.random.permutation(np.where(colors >= minmaxcolor)[0])[:3000]
            posdensity = np.log(1+gaussian_kde(vals[:, posmask], weights = densweight[posmask])(vals))
            negmask = np.random.permutation(np.where(colors <= minmaxcolor)[0])[:3000]
            negdensity = np.log(1+gaussian_kde(vals[:, negmask], weights = densweight[negmask])(vals))
        else:
            posmask = np.random.permutation(np.where(colors >= minmaxcolor)[0])[:3000]
            posdensity = gaussian_kde(vals[:, posmask], weights = densweight[posmask])(vals)
            negmask = np.random.permutation(np.where(colors <= minmaxcolor)[0])[:3000]
            negdensity = gaussian_kde(vals[:, negmask], weights = densweight[negmask])(vals)
            
        posdensity = posdensity[sortd]
        negdensity = negdensity[sortd]
        ax.tricontour(vals[0][sortd], vals[1][sortd], posdensity, levels=6, linewidths=0.5, colors = [cm.get_cmap(cmap)(1.)])
        ax.tricontour(vals[0][sortd], vals[1][sortd], negdensity, levels=6, linewidths=0.5, colors = [cm.get_cmap(cmap)(0.)])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    lim = [np.amax(np.amin(vals,axis = 1)), np.amin(np.amax(vals,axis = 1))]
    
    if '--plotdiagonal' in sys.argv:
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color = 'maroon')
    if '--plotnegdiagonal' in sys.argv:
        ax.plot([lim[0], lim[1]], [lim[1], lim[0]], color = 'maroon')
    
    ax.set_xlabel(namex)
    ax.set_ylabel(namey)
    ax.legend()
    
    if '--annotate_header' in sys.argv and header is not None:
        for h, head in enumerate(header):
            ax.text(vals[0][h], vals[1][h], head , ha = 'left', va = 'bottom')
    
    if '--colorfile' in sys.argv:
        pax = ax.get_position()
        ax2 = fig.add_subplot(555)
        print(pax.x0, pax.width, pax.y0, pax.height)
        ax2.set_position([pax.x0+pax.width, pax.y0+pax.height*5/6, pax.width * 1/12., pax.height*1/6.])
        ax2.imshow(np.linspace(0,1,100)[::-1].reshape(-1,1), cmap = cmap, vmin = 0, vmax = 1, aspect= 'auto') 
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
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if '--zeroxaxis' in sys.argv:
        ax.plot(xlim, [0,0], color = 'grey')
    if '--zeroyaxis' in sys.argv:
        ax.plot([0,0], ylim, color = 'grey')
    
    if '--logx' in sys.argv:
        ax.set_xscale('log')
    if '--symlogx' in sys.argv:
        ax.set_xscale('symlog')
    if '--logy' in sys.argv:
        ax.set_yscale('log')
    if '--symlogy' in sys.argv:
        ax.set_yscale('symlog')
    
    
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
        
