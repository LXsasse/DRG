import numpy as np
import matplotlib.pyplot as plt
import sys,os
from scipy.linalg import svd
from drg_tools.io_utils import read_matrix_files as read_file
from drg_tools.plotlib import vulcano 

if __name__ == '__main__':

    delimiter = ','
    if '--delimiter' in sys.argv:
        delimiter = sys.argv[sys.argv.index('--delimiter')+1]
    ## read FC and p-value files
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
    # generates vulcano plots for all experiments in matrix file
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

    


