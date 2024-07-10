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
        else:
            header = line.strip('#').strip().split(delimiter)
    return np.array(names), np.array(vals,dtype = float), header


if __name__ == '__main__':    
    names, scores, header = readfiles(sys.argv[1], ' ')
    pcheck, pcut = 1, 0.5
    trreg = 3
    ptrreg = 5
    
    mask = scores[:,pcheck] < pcut
    names, scores = names[mask], scores[mask]
    
    
    fnames, fcvals, experiments = readfiles(sys.argv[2], ',')
    mask = np.isin(fnames,names)
    haspos = np.sum(fcvals[mask] > 0,axis = 1) > 0
    hasneg = np.sum(fcvals[mask] < 0,axis = 1) > 0
    haspos = haspos * hasneg
    fnames, fcvals = fnames[mask], np.sum(fcvals[mask], axis = 1)#p.sum(np.absolute(fcvals[mask]), axis = 1)

    
    fig = plt.figure(figsize = (3.5,3.5), dpi = 200)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(scores[~haspos,ptrreg], scores[~haspos,trreg], s = 50*(1-scores[~haspos,pcheck]), vmin = -5, vmax = 5, alpha = 0.5, cmap = 'coolwarm', c = fcvals[~haspos], label ='single-directional effect')
    ax.scatter(scores[haspos,ptrreg], scores[haspos,trreg], s = 50*(1-scores[haspos,pcheck]), vmin = -5, vmax = 5, alpha = 0.5, cmap = 'coolwarm', c = fcvals[haspos], marker = 's', label = 'up and down\nregulated')
    
    lim = [np.amin(scores[:,[ptrreg,trreg]],axis = 0), np.amax(scores[:,[ptrreg,trreg]],axis = 0)]
    ax.plot([0, lim[1][0]], [0, lim[1][1]], color = 'grey')
    ax.legend(fontsize = 6)
    #ax.plot([0,0],[lim[0][1], lim[1][1]], color = 'grey', ls = '--')
    #ax.plot([lim[0][0], lim[1][0]],[0,0], color = 'grey', ls = '--')
    ax.set_xlabel('Post-transcriptional')
    ax.set_ylabel('Transcriptional')
    
    pax = ax.get_position()
    ax2 = fig.add_subplot(555)
    ax2.set_position([pax.x0+pax.width, pax.y0+pax.height*2/3, 0.05, pax.height*1/3])
    ax2.imshow(np.linspace(0,1,100)[::-1].reshape(-1,1), cmap = 'coolwarm', vmin = 0, vmax = 1, aspect= 'auto') 
    ax2.set_yticks([0,50,100])
    ax2.set_yticklabels(['>5','0','<-5'])
    ax2.tick_params(left = False, labelleft = False, right = True, labelright = True, bottom = False, labelbottom = False)
    
    if '--savefig' in sys.argv:
        outname = sys.argv[sys.argv.index('--savefig')+1]
        fig.savefig(outname+'_scatter.svg', dpi = 400, bbox_inches = 'tight')
    else:
        plt.show()
        
