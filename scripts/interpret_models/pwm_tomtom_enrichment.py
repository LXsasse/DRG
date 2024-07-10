import numpy as np
import sys, os
import matplotlib.pyplot as plt 
import scipy.stats as sct

def readtomtom(f):
    obj = open(f,'r').readlines()
    names = []
    pvals =[]
    qvals = []
    for l, line in enumerate(obj):
        if l > 0 and line[0] != '#':
            line = line.strip().split('\t')
            if len(line) > 5:
                names.append(line[0])
                pvals.append(line[3])
                qvals.append(line[5])
        
    names = np.array(names)
    pvals = np.array(pvals, dtype = float)
    qvals = np.array(qvals, dtype = float)
    return names, pvals, qvals


def boxplot(x, xticklabels = None, xticks = None, ylabel = None, stats_test = True):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.boxplot(x)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xticklabels is not None:
        if xticks is None:
            xticks = np.arange(len(xticklabels))
        ax.set_xticklabels(xticklabels, rotation = 60)
    
    if stats_test:
        for i, x0 in enumerate(x):
            for j, x1 in enumerate(x):
                if j > i and i != j:
                    wst, pst = sct.ranksums(x0, x1, alternative='greater')
                    print(i, j, pst)
    return fig

def barplot(y, xticklabels = None, xticks = None, width = 0.8, ylabel = None, stats_test = True, color = None, cmap = None):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    x = np.arange(len(y))+0.5
    y = np.array(y).flatten()
    
    if len(y) > len(x):
        di = int(len(y)/len(x))
        x = np.repeat(x, di)
        unit = width/(di+1)
        new = (np.arange((1.-width)/2, 1. - width/2, unit) - 0.5).reshape(1,-1)
        new = np.repeat(new, axis = 0).flatten()
        x = x + new
        width = width/di
        if cmap is not None:
            color = plt.get_cmap(cmap)(np.linspace(0,1.,len(new)))
    ax.bar(x,y, color = color, width = width)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if xticklabels is not None:
        if xticks is None:
            xticks = np.arange(len(xticklabels))+0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation = 60)
        
    return fig


def scatterplot(x, y, xlabel = None, ylabel = None, color = None):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(x, y, c = color, cmap = 'Purples')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return fig


if __name__ == "__main__":
    
    tnames, tpvals, tqvals = readtomtom(sys.argv[1])
    
    names = np.unique(tnames)
    print(len(names))
    
    maxq = np.zeros(len(names))
    maxp = np.zeros(len(names))
    for n, name in enumerate(names):
        maxq[n] = -np.log10(np.amin(tqvals[tnames == name]))
        maxp[n] = -np.log10(np.amin(tpvals[tnames == name]))
    
    if '--compare' in sys.argv:
        tnames, tpvals, tqvals = readtomtom(sys.argv[sys.argv.index('--compare')+1])
    
    if '--cutoff' in sys.argv:
        value = sys.argv[sys.argv.index('--cutoff')+1]
        cutoff = float(sys.argv[sys.argv.index('--cutoff')+2])
        if value == 'q':
            mask = maxq <= cutoff
        else:
            mask = maxp <= cutoff
        names, maxq, maxp = names[mask], maxq[mask], maxp[mask]
    
    
    if '--clusterfile' in sys.argv:
        clusters = np.genfromtxt(sys.argv[sys.argv.index('--clusterfile')+1], dtype = str)
        seeds = np.array([f.split('_')[-1] for f in clusters[:,0]])
        cnns = np.array([f.split('_')[1] for f in clusters[:,0]])
        nseed = []
        fraccnn = []
        for n, name in enumerate(names):
            filters = clusters[:,1] == name
            nseed.append(len(np.unique(seeds[filters])))
            fraccnn.append( np.sum(cnns[filters] == 'CNN0')/np.sum(filters) )
        nseed = np.array(nseed)
        fraccnn = np.array(fraccnn)
        x = [maxp[nseed == n] for n in np.unique(nseed)]
        fig = boxplot(x, xticklabels = np.unique(nseed))
        #figs = scatterplot(fraccnn, maxp, xlabel = 'Fraction DNA', ylabel = 'logP match', color = nseed)
        #plt.show()
    
    else:
        cnn=np.array([n.split('_')[1] for n in names])
        x = [maxp[cnn == 'CNN0'], maxp[cnn == 'CNN1']]
        if '--boxplot' in sys.argv:
            fig = boxplot(x, xticklabels = ['DNA-CNN', 'RNA-CNN'])
        else:
            x = [len(xi) for xi in x]
            fig = barplot(x, xticklabels = ['DNA-CNN', 'RNA-CNN'], color = 'grey')
        plt.show()
    
    
    
