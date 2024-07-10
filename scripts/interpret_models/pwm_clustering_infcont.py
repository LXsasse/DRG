
import numpy as np
import sys, os
import matplotlib.pyplot as plt 

def histogram(x, nbins = None, xlabel = None, logy = False):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if nbins is None:
        nbins = np.linspace(-0.5, int(np.amax(x))+0.5, int(np.amax(x))+2)
        
    ax.hist(x, bins = nbins, color = 'Indigo', alpha = 0.6, histtype = 'bar', lw = 1 )
    ax.hist(x, bins = nbins, color = 'Indigo', alpha = 1., histtype = 'step' )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale('log')
    return fig

def scatterplot(x, y, xlabel = None, ylabel = None, color = None, size = None, cmap = None, textnum = False, colornum = False, wigglex = 0):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if color is not None:
        sort = np.argsort(color)
        x, y, color = x[sort], y[sort], color[sort]
        if size is not None:
            size = size[sort]
    if wigglex > 0:
        x = x+(np.random.random(len(x))-0.5)*wigglex
    if colornum:
        upos, npos = np.unique(np.array([x,y]).T, axis = 0, return_counts = True)
        ax.scatter(upos[:,0], upos[:,1], cmap = cmap, s = size, c = npos, alpha = 0.5)
    else:
        ax.scatter(x, y, cmap = cmap, s = size, c = color)
    if textnum:
        upos, npos = np.unique(np.array([x,y]).T, axis = 0, return_counts = True)
        for u, up in enumerate(upos):
            ax.text(up[0], up[1], str(npos[u]), ha = 'left', va = 'bottom')
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    return fig

def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool

def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split('\t')
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float)
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
    return pwms, names   

def info_content(pwm):
    ic = np.mean(np.sum(-pwm*np.log2((pwm+1e-16)/0.25) , axis = 1))
    return ic
    

if __name__ == '__main__':
    clusterfile = sys.argv[1]
    
    pwms, pnames = read_pwm(clusterfile)
    
    infocontent = []
    clusters = []
    names = []
    for p, pwm in enumerate(pwms):
        infocontent.append(info_content(pwm))
        names.append(pnames[p].split(';'))
        clusters.append([p for i in range(len(names[-1]))])
        
    infocontent = np.array(infocontent)
    names = np.concatenate(names)
    clusters = np.concatenate(clusters)
    
    
    seeds = np.array([n.split('_')[-1] for n in names])
    cnns = np.array([n.split('_')[1] for n in names])

    ucluster, ncluster = np.unique(clusters, return_counts = True)
    nseed, fsub = np.zeros(len(ucluster)), np.zeros(len(ucluster))
    for c, clust in enumerate(ucluster):
        mask = clusters == clust
        nseed[c] = len(np.unique(seeds[mask]))
        fsub[c] = np.sum(cnns[mask] == 'CNN0')/ncluster[c]
    

# Information content distribution 
    fig1 = histogram(infocontent, xlabel = 'Information content', logy = True, nbins = 21)
# Number of reproducable versus IC

    pure = np.copy(fsub)
    pure[pure>0.5] = 1-pure[pure>0.5]
    pure = 0.5-pure
    fig2 = scatterplot(nseed, infocontent, xlabel = 'Number of seeds with PWM', ylabel = 'Information Content', color = pure)
# 
    fig3 = scatterplot(infocontent, fsub, color = nseed, xlabel = 'Information Content', ylabel = 'Number of PWMs in CNN0')
    
    fig1.savefig(os.path.splitext(clusterfile)[0]+'.ICdist.jpg', dpi = 300, bbox_inches='tight')
    fig2.savefig(os.path.splitext(clusterfile)[0]+'.NsvsIC.jpg', dpi = 300, bbox_inches='tight')
    fig3.savefig(os.path.splitext(clusterfile)[0]+'.ICvsfs.jpg', dpi = 300, bbox_inches='tight')
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()
