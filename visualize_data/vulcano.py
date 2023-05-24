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

def vulcano(fc, pv, figname, xcutoff=1., ycutoff = 1.96, text = None, mirror = False, eigenvalues = False, colors = None, cmap = 'viridis'):
    fig = plt.figure(figname, figsize = (4,4), dpi = 200)
    ax = fig.add_subplot(111)
    maskx = np.absolute(fc) > xcutoff
    masky = pv > ycutoff
    if mirror:
        pv = np.sign(fc) * pv
    if colors is not None:
        sort = np.argsort(colors)
        ap = ax.scatter(fc[sort], pv[sort], cmap = cmap, c = colors[sort], alpha = 0.6)
        fig.colorbar(ap, aspect = 2, pad = 0, anchor = (0,0.9), shrink = 0.15)
    else:
        ax.scatter(fc[~(maskx*masky)], pv[~(maskx*masky)], c = 'grey', alpha = 0.3)
        ax.scatter(fc[maskx*masky], pv[maskx*masky], c='maroon', alpha = 0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(figname)
    ax.set_xlabel('Log2 Fold change')
    ax.set_ylabel('Log10 p-value')
    if mirror:
        ax.set_ylabel('Log10 sign p-value')
    if eigenvalues:
        u, s, v = svd(np.array([fc,pv]), full_matrices=False, compute_uv=True)
        u = u/u[0]
        ax.plot([-u[0,0],0, u[0,0]], [-u[1,0], 0, u[1,0]], color = 'r', ls = '--', label = 'Eigval1 (m='+ str(np.around(u[1,0]/u[0,0],2))+ ')')
        ax.legend()
        
    
    if text is not None:
        for s in np.where(masky*maskx)[0]:
            if fc[s] < 0:
                ax.text(fc[s], pv[s], text[s], ha = 'left', size = 8)
            else:
                ax.text(fc[s], pv[s], text[s],ha = 'right', size = 8)
    fig.tight_layout()
    return fig

delimiter = ','
if '--delimiter' in sys.argv:
    delimiter = sys.argv[sys.argv.index('--delimiter')+1]

expfc, genfc, valfc = read_file(sys.argv[1], expdelim = '.FC', delimiter = delimiter, nan_value = '0')
exppv, genpv, valpv = read_file(sys.argv[2], expdelim = '.PV', delimiter = delimiter, nan_value = '1')
valpv[np.isnan(valpv)] = 1

if not np.array_equal(genfc, genpv):
    print("Genes different", genfc, genpv)
    sys.exit()
if not np.array_equal(expfc, exppv):
    print("Exp different", expfc, exppv)
    sys.exit()

print(len(genfc), len(expfc), np.shape(valfc), np.shape(valpv))

add = ''
if '--outname' in sys.argv:
    add = sys.argv[sys.argv.index('--outname')+1]    

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

colors = [None for e in range(len(expfc))]
if '--colorfile' in sys.argv:
    colorfile = sys.argv[sys.argv.index('--colorfile')+1]
    ccol = np.array(sys.argv[sys.argv.index('--colorfile')+2].split(','), dtype = int)
    cdelimiter = None
    if '--colordelimiter' in sys.argv:
        cdelimiter = sys.argv[sys.argv.index('--colordelimiter')+1]
    header, cnames, colors = read_file(colorfile, delimiter = cdelimiter)
    print('Color column', header[ccol])
    colors = colors[:,ccol]
    sort = np.argsort(cnames)[np.isin(cnames, genfc)]
    cnames, colors = cnames[sort], colors[sort]
    colors = colors.T
    print(np.shape(colors))
    if not np.array_equal(cnames, genfc):
        print('Colors are incomplete. Only present', int(np.sum(np.isin(cnames, genfc))))
        sys.exit()
    

#print expfc, exppv
#print np.array_equal(genfc, genpv)
#print np.array_equal(expfc, exppv)

eigvals = False
mirror = False
if '--mirror' in sys.argv:
    mirror = True
    add += '_mirror'
    if '--eigenvalues' in sys.argv:
        eigvals = True
        add += '_eigval'

if '--onlylist' in sys.argv:
    foldcut = float(sys.argv[sys.argv.index('--onlylist')+1])
    pcut = float(sys.argv[sys.argv.index('--onlylist')+2])

siggenes = []
for e, exp in enumerate(expfc):
    d = list(exppv).index(exp)
    if '--onlylist' in sys.argv:
        sigs = np.where((np.absolute(valfc[:,e]) > foldcut) & (valpv[:,d]>pcut))[0]
        siggenes.append(sigs)
        print(exp+add+'_fc'+str(foldcut)+'_p'+str(pcut)+'.txt')
        print(exp, len(sigs))
        np.savetxt(exp+add+'_fc'+str(foldcut)+'_p'+str(pcut)+'.txt', genpv[sigs].reshape(-1,1), fmt = '%s')
    else:
        fi = vulcano(valfc[:,e], valpv[:,d], exp, mirror = mirror, eigenvalues = eigvals, colors = colors[e]) #, text = genfc)
        if '--savefig' in sys.argv:
            print(exp+add+'_vulcano.jpg')
            fi.savefig(exp+add+'_vulcano.jpg', dpi = 200, bbox_inches='tight')
        else:
            fi.tight_layout()
            plt.show()
        plt.close()


if '--combined_stats' in sys.argv:
    ctypes = []
    Ils = []
    for e, exp in enumerate(expfc):
        ctypes.append(exp.split('_')[0])
        Ils.append(exp.split('_')[1])
    unctypes = np.unique(ctypes)
    unils = np.unique(Ils)
    sigmatrix = np.zeros((len(unctypes), len(unils)))
    for u, unct in enumerate(unctypes):
        for w, uil in enumerate(unils):
            sigmatrix[u,w] = len(siggenes[list(expfc).index(unct+'_'+uil[2:])])

    fig = plt.figure(figsize = (4,4), dpi = 200)
    ax = fig.add_subplot(111)
    ax.imshow(np.log(sigmatrix), aspect='auto')
    ax.set_xticks(np.arange(len(unils)))
    ax.set_xticklabels(unils)
    ax.set_yticks(np.arange(len(unctypes)))
    ax.set_yticklabels(unctypes)
    for u, unct in enumerate(unctypes):
        for w, uil in enumerate(unils):
            ax.text(w, u, str(int(sigmatrix[u,w])), ha = 'center', va = 'center')
    fig.savefig('Significant_FC'+str(foldcut)+'_PV'+str(pcut)+'.jpg', dpi = 250)

    for e, exp in enumerate(expfc):
        print( exp.replace('_', '.'), len(siggenes[e]))

    siggenes = np.unique(np.concatenate(siggenes))
    print( len(siggenes))
    obj = np.savetxt('Significant_FC'+str(foldcut)+'_PV'+str(pcut)+'.list', genfc[siggenes], fmt = '%s')

    


