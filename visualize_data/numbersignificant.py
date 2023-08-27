import numpy as np
import matplotlib.pyplot as plt
import sys,os
from scipy.linalg import svd

def read_file(f, expdelim = None, delimiter = ',', nan = 'NA', nan_value = 'nan'):
    f = open(f, 'r').readlines()
    experiments = f[0].strip('#').strip().replace('"','').split(delimiter)
    if expdelim is not None:
        for e, exp in enumerate(experiments):
            experiments[e] = exp.replace(expdelim, '_')
    genes = []
    values = []
    for l, line in enumerate(f):
        if l >= 1:
            line = line.strip().split(delimiter)
            genes.append(line[0].strip('"'))
            ival = np.array(line[1:])
            ival[ival == nan] = nan_value
            values.append(ival)
    return np.array(experiments), np.array(genes), np.array(values, dtype = float)


def heatmap(mat, cmap = 'Purples', text=True, log_colors = False, xlabels = None, ylabels = None, limit = None):
    fig = plt.figure(figsize=(0.3*np.shape(mat)[1],0.3*np.shape(mat)[0]),dpi = 200)
    ax = fig.add_subplot(111)
    pmat = np.copy(mat)
    if log_colors:
        pmat = np.log(pmat)
    
    if limit is None:
        limit = np.amax(pmat)
    
    ax.imshow(pmat, aspect = 'auto', cmap = cmap, vmin = 0, vmax = limit)
    
    if text:
        for i in range(np.shape(mat)[0]):
            for j in range(np.shape(mat)[1]):
                if pmat[i,j] > limit*2/3:
                    tc = 'silver'
                else:
                    tc = 'k'
                ax.text(j,i,str(mat[i,j]), va = 'center', ha = 'center', fontsize = 6, color = tc)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation = 45)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    return fig

delimiter = '\t'
if '--delimiter' in sys.argv:
    delimiter = sys.argv[sys.argv.index('--delimiter')+1]

expfc, genfc, valfc = read_file(sys.argv[1], expdelim = '.FC', delimiter = delimiter, nan_value = '0')
exppv, genpv, valpv = read_file(sys.argv[2], expdelim = '.PV', delimiter = delimiter, nan_value = '1')
valpv[np.isnan(valpv)] = 1

outname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]

foldcut = float(sys.argv[3])
pcut = float(sys.argv[4]) 

outname += 'fc'+str(foldcut)+'pv'+str(pcut)

if not np.array_equal(genfc, genpv):
    print("Genes different", genfc, genpv)
    sys.exit()
if not np.array_equal(expfc, exppv):
    print("Exp different", expfc, exppv)
    sys.exit()
    
print(len(genfc), len(expfc), np.shape(valfc), np.shape(valpv))

experiments = np.array([e.rsplit('_',2)[0].split('_') for e in expfc])
cells = np.unique(experiments[:,1])
ils = np.unique(experiments[:,0])
isort = np.array([il.strip('IL') for il in ils], dtype = int)
ils = ils[np.argsort(isort)]
genes = genfc

add = ''
if '--adjustp' in sys.argv:
    add += '.bhadj'
    from statsmodels.stats.multitest import multipletests
    for v in range(len(exppv)):
        issig, padj, a_, b_ = multipletests(valpv[:,v], method = 'fdr_bh')
        valpv[:,v] = padj

if '--logp' in sys.argv:
    add += '.logp10'
    valpv = -np.log10(valpv)
    valpv[np.isnan(valpv)] = 0
    
if '--logfc' in sys.argv:
    add += '.logfc2'
    valfc = np.log2(valfc)    


if '--splitdirection' in sys.argv:
    sigmat = np.zeros((2,len(ils), len(cells)), dtype = int)
else:
    sigmat = np.zeros((len(ils), len(cells)), dtype = int)

if '--negate' in sys.argv:
    print('Flip FC sign')
    valfc = -valfc

updownlists = [[],[],[]]
for c, cell in enumerate(cells):
    emask = experiments[:,1] == cell
    for i, il in enumerate(ils):
        mask = np.where(emask * (experiments[:,0] == il))[0]
        if len(mask) > 0:
            mask = mask[0]
            sigs = (np.absolute(valfc[:,mask]) > foldcut) & (valpv[:,mask]>pcut)
            print(cell, il, int(np.sum(sigs)))
            up = sigs*valfc[:,mask] > foldcut
            down = sigs*valfc[:,mask] < -foldcut
            if '--splitdirection' in sys.argv:
                sigmat[0,i,c] = int(np.sum(up))
                sigmat[1,i,c] = int(np.sum(down))
            else:
                sigmat[i,c] = int(np.sum(sigs))
            updownlists[0].append([cell, il])
            updownlists[1].append(genes[up])
            updownlists[2].append(genes[down])

np.savez_compressed(outname + add+'updownlists.npz', lists = updownlists)
        
        
limit = None
if '--limit' in sys.argv:
    limit = int(sys.argv[sys.argv.index('--limit')+1])

if '--splitdirection' in sys.argv:
    if limit is None:
        limit = np.amax(sigmat)
    fig0 = heatmap(sigmat[0], cmap = 'Reds', xlabels = cells, ylabels = ils, limit = limit)
    fig1 = heatmap(sigmat[1], xlabels = cells, ylabels = ils, limit = limit)
    fig0.savefig(outname + add+'_up.jpg', dpi = 250, bbox_inches = 'tight')
    fig1.savefig(outname + add+'_down.jpg', dpi = 250, bbox_inches = 'tight')
else:
    fig = heatmap(sigmat, xlabels = cells, ylabels = ils, limit = limit)
    fig.savefig(outname + add+'.jpg', dpi = 250, bbox_inches = 'tight')



plt.show()
    
    
    
