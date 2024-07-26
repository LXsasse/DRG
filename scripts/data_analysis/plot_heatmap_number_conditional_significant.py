
import numpy as np
import matplotlib.pyplot as plt
import sys,os

from drg_tools.plotlib import _plot_heatmap as heatmap
'''
Use statistics of up and down regulated genes in 3 different conditions to determine overlaps between groups.
TODO
make general to different conditions. 
'''

if __name__ == '__main__':
    # specific npz files that contained the names of data points that are
    #significant, and a split between up and down
    expsig = np.load(sys.argv[1], allow_pickle = True)['lists']
    ktsig = np.load(sys.argv[2], allow_pickle = True)['lists']
    kdsig = np.load(sys.argv[3], allow_pickle = True)['lists']

    outname = sys.argv[4]

    ext, expup, expdown = expsig
    ktt, ktup, ktdown = ktsig
    kdt, kdup, kddown = kdsig
    ext = np.array(list(ext))
    ktt = np.array(list(ktt))
    kdt = np.array(list(kdt))

    cells = np.unique(np.concatenate([ext[:,0], ktt[:,0], kdt[:,0]]))
    ils = np.unique(np.concatenate([ext[:,1], ktt[:,1], kdt[:,1]]))

    isort = np.array([il.strip('IL') for il in ils], dtype = int)
    ils = ils[np.argsort(isort)]

    stats = ['Exp_by_KT', 'Exp_by_Kd', 'Exp_by_KTandKD', 'Kd_oppsing_KT', 'KT_oppsing_Kd']
    counts = np.zeros((4,2,len(ils), len(cells)), dtype = int)
    opcounts = np.zeros((2,len(ils), len(cells)), dtype = int)
    for c, cell in enumerate(cells):
        for i, il in enumerate(ils):
            emask = np.where((ext[:,0] == cell) * (ext[:,1] == il))[0]
            ktmask = np.where((ktt[:,0] == cell) * (ktt[:,1] == il))[0]
            kdmask = np.where((kdt[:,0] == cell) * (kdt[:,1] == il))[0]
            if len(emask) > 0:
                hkt, hkd = False, False
                emask = emask[0]
                eup,edo = expup[emask], expdown[emask]
                allex = np.union1d(eup, edo)
                allcolor = np.zeros(len(allex), dtype = int)
                if len(ktmask) > 0:
                    hkt = True
                    ktmask = ktmask[0]
                    tup,tdo = ktup[ktmask], ktdown[ktmask]
                    ebytup = np.intersect1d(eup, tup)
                    ebytdo = np.intersect1d(edo, tdo)
                if len(kdmask) > 0:
                    hkd = True
                    kdmask = kdmask[0]
                    dup,ddo = kdup[kdmask], kddown[kdmask]
                    ebydup = np.intersect1d(eup, dup)
                    ebyddo = np.intersect1d(edo, ddo)
                if hkt and hkd:
                    ebydtup = np.intersect1d(ebytup, ebydup)
                    ebytup = np.setdiff1d(ebytup, ebydtup)
                    ebydup = np.setdiff1d(ebydup, ebydtup)
                    ebydtdo = np.intersect1d(ebytdo, ebyddo)
                    ebytdo = np.setdiff1d(ebytdo, ebydtdo)
                    ebyddo = np.setdiff1d(ebyddo, ebydtdo)
                    counts[2,0,i,c] = len(ebydtup)
                    counts[2,1,i,c] = len(ebydtdo)
                    ktupkddo = np.intersect1d(tup, ddo)
                    kdupktdo = np.intersect1d(tdo, dup)
                    opcounts[0,i,c] = len(ktupkddo)
                    opcounts[1,i,c] = len(kdupktdo)
                    allcolor[np.isin(allex, np.append(ktupkddo,kdupktdo))] = 4
                    allcolor[np.isin(allex, np.append(ebydtup,ebydtdo))] = 3
                    
                counts[0,0,i,c] = len(ebytup)
                counts[0,1,i,c] = len(ebytdo)
                counts[1,0,i,c] = len(ebydup)
                counts[1,1,i,c] = len(ebyddo)
                counts[3,0,i,c] = len(eup) - np.sum(counts[:3,0,i,c])
                counts[3,1,i,c] = len(edo) - np.sum(counts[:3,1,i,c])
                allcolor[np.isin(allex, np.append(ebytup,ebytdo))] = 1
                allcolor[np.isin(allex, np.append(ebydup,ebyddo))] = 2
                
                np.savetxt(cell+'_'+il+'_type_colors.txt', np.array([allex, allcolor.astype(str)]).T, fmt = '%s')
                print('Saved', cell+'_'+il+'_type_colors.txt')
            
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit')+1])

    if limit is None:
        limit = np.amax(np.sum(counts, axis = 0))
    figtu = heatmap(counts[0,0], cmap = 'Reds', xlabel = cells, ylabel = ils, vlim = limit, title = 'Up ex by KT')
    figtd = heatmap(counts[0,1], cmap = 'Purples', xlabel = cells, ylabel = ils, vlim  = limit, title = 'Down ex by KT')
    figdu = heatmap(counts[1,0], cmap = 'Reds', xlabel = cells, ylabel = ils, limit = limit, title = 'Up ex by Stability')
    figdd = heatmap(counts[1,1], cmap = 'Purples',xlabel = cells, ylabel = ils, vlim  = limit, title = 'Down ex by Stability')
    figtdu = heatmap(counts[2,0], cmap = 'Reds', xlabel = cells, ylabel = ils, limit = limit, title = 'Up ex by KT and Stab.')
    figtdd = heatmap(counts[2,1], cmap = 'Purples',xlabel = cells, ylabel = ils, vlim  = limit, title = 'Down ex by KT and Stab')

    figup = heatmap(counts[3,0], cmap = 'Reds', xlabel = cells, ylabel = ils, limit = limit, title = 'Up ex non-sig comb.')
    figdo = heatmap(counts[3,1], cmap = 'Purples',xlabel = cells, ylabel = ils, vlim  = limit, title = 'Down ex non-sig comb.')

    figtudd = heatmap(opcounts[0], cmap = 'Greys', xlabel = cells, ylabel = ils, limit = limit, title = 'Up KT and down Kd')
    figtddu = heatmap(opcounts[1], cmap = 'Greys', xlabel = cells, ylabel = ils, vlim  = limit, title = 'Up Kd and down KT')

    figtu.savefig(outname + '_upKT.jpg', dpi = 250, bbox_inches = 'tight')
    figtd.savefig(outname + '_downKT.jpg', dpi = 250, bbox_inches = 'tight')
    figdu.savefig(outname + '_upStab.jpg', dpi = 250, bbox_inches = 'tight')
    figdd.savefig(outname + '_downStab.jpg', dpi = 250, bbox_inches = 'tight')
    figtdu.savefig(outname + '_upKTStab.jpg', dpi = 250, bbox_inches = 'tight')
    figtdd.savefig(outname + '_downKTStab.jpg', dpi = 250, bbox_inches = 'tight')
    figup.savefig(outname + '_upnone.jpg', dpi = 250, bbox_inches = 'tight')
    figdo.savefig(outname + '_downnone.jpg', dpi = 250, bbox_inches = 'tight')
    figtudd.savefig(outname + '_KTupStabdown.jpg', dpi = 250, bbox_inches = 'tight')
    figtddu.savefig(outname + '_KTdownStabup.jpg', dpi = 250, bbox_inches = 'tight')

    plt.show()



