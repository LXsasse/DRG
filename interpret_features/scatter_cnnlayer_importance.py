#scatter_pv_fc.py
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce
import glob
import seaborn as sns
from matplotlib import cm
from scipy.stats import skew

def readfiles(file, delimiter):
    fi = open(file, 'r').readlines()
    names, vals,header = [], [], None
    for l, line in enumerate(fi):
        if line[0] != '#':
            line = line.strip().split(delimiter)
            names.append(line[0])
            vals.append(line[1:])
        elif l == 0:
            header = line.strip('#').strip().split(delimiter)
    
    names, vals = np.array(names), np.array(vals,dtype = float)
    sort = np.argsort(names)
    names, vals = names[sort], vals[sort]
    return names, vals, np.array(header)


if __name__ == '__main__': 
    # _netattX.dat file
    names, scores, header = readfiles(sys.argv[1], ' ')
    
    pcheck, pcut = 1, 0.5 # column with confidence values and cutoff 0: MSE, 1: Corr
    trreg = 3 # column for CNN0, 2: DMSE, 3: DCorr
    ptrreg = 5 # column for CNN1, 4: DMSE, 5: DCorr
    
    # select only confident predictions
    mask = scores[:,pcheck] < pcut
    names, scores = names[mask], scores[mask]
    
    colored = False
    if '--color_class' in sys.argv:
        colored = True
        # Class file to color each gene according to its direction of significant changes
        fnames, fcvals, experiments = readfiles(sys.argv[sys.argv.index('--color_class')+1], ',')
        ctype = sys.argv[sys.argv.index('--color_class')+2]
        typemask = [ctype in t for t in experiments]
        fcvals = fcvals[:,typemask]
        mask = np.isin(fnames,names)
        haspos = np.sum(fcvals[mask] > 0,axis = 1) > 0
        hasneg = np.sum(fcvals[mask] < 0,axis = 1) > 0
        haspos = haspos * hasneg
        fnames, fcvals = fnames[mask], np.sum(fcvals[mask], axis = 1)
        mask = np.isin(names,fnames)
        names, scores = names[mask], scores[mask]
        
        if '--significant_only' in sys.argv:
            mask = haspos | (fcvals != 0)
            fnames, fcvals, names, scores, haspos, hasneg = fnames[mask], fcvals[mask], names[mask], scores[mask], haspos[mask], hasneg[mask]
    
    elif '--color_scale' in sys.argv:
        colored = True
        # Class file to color each gene according to its direction of significant changes
        fnames, fcvals, experiments = readfiles(sys.argv[sys.argv.index('--color_scale')+1], ' ')
        fcvals = fcvals[:,-1]
        mask = np.isin(fnames,names)
        fnames, fcvals = fnames[mask], fcvals[mask]
        mask = np.isin(names,fnames)
        names, scores = names[mask], scores[mask]
        haspos = np.ones(len(names)) ==0
        
        if '--significant_only' in sys.argv:
            signfile = np.genfromtxt(sys.argv[sys.argv.index('--significant_only')+1], dtype = str)
            mask = np.isin(names, signfile)
            fnames, fcvals, names, scores, haspos = fnames[mask], fcvals[mask], names[mask], scores[mask], haspos[mask]
        
    else:
        fcvals = np.ones(len(names))*-4
        haspos = np.ones(len(names)) ==0
    
    
    fig = plt.figure(figsize = (3.5,3.5), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    mabs = int(np.amax(np.absolute(fcvals)))
    if np.amin(fcvals) < 0:
        vmin, vmax = max(-5,-mabs), min(5,mabs)
    else:
        vmin, vmax = -mabs, mabs # int(np.floor(np.amin(fcvals))), int(np.ceil(mabs))
    
    sort = np.argsort(np.absolute(fcvals[~haspos]))
    ax.scatter(scores[~haspos,ptrreg][sort], scores[~haspos,trreg][sort], s = 50*(1-scores[~haspos,pcheck][sort]), vmin = vmin, vmax = vmax, alpha = 0.5, cmap = 'coolwarm', c = fcvals[~haspos][sort], label ='unique-directional effect')
    
    if '--print_values' in sys.argv:
        for i in range(len(scores[~haspos,ptrreg][sort])):
            print(scores[~haspos,ptrreg][i], scores[~haspos,trreg][i], fcvals[~haspos][i])
        
    
    sort = np.argsort(np.absolute(fcvals[haspos]))
    ax.scatter(scores[haspos,ptrreg][sort], scores[haspos,trreg][sort], s = 50*(1-scores[haspos,pcheck][sort]), vmin = vmin, vmax = vmax, alpha = 0.5, cmap = 'coolwarm', c = fcvals[haspos][sort], marker = 's', label = 'up and down\nregulated')
    
    lim = [np.amin(scores[:,[ptrreg,trreg]],axis = 0), np.amax(scores[:,[ptrreg,trreg]],axis = 0)]
    ax.plot([0, lim[1][0]], [0, lim[1][1]], color = 'grey')
    ax.legend(fontsize = 6)
    
    if '--axislabel' in sys.argv:
        xlabel = sys.argv[sys.argv.index('--axislabel')+1]
        ylabel = sys.argv[sys.argv.index('--axislabel')+2]
    else:
        xlabel ='Post-transcriptional'
        ylabel ='Transcriptional'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if colored:
        # define a colorbar
        pax = ax.get_position()
        ax2 = fig.add_subplot(555)
        ax2.set_position([pax.x0+pax.width, pax.y0+pax.height*2/3, 0.05, pax.height*1/3])
        ax2.imshow(np.linspace(0,1,100)[::-1].reshape(-1,1), cmap = 'coolwarm', vmin = 0, vmax = 1, aspect= 'auto') 
        
        ax2.set_yticks([0,50,100])
        ax2.set_yticklabels(['>'+str(vmax), str((vmin+vmax)/2),'<'+str(vmin)])
        ax2.tick_params(left = False, labelleft = False, right = True, labelright = True, bottom = False, labelbottom = False)
    
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        fig.savefig(outname+'_scatter.jpg', dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()
        
