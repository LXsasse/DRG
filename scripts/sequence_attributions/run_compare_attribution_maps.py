import numpy as np
import sys, os
import matplotlib.pyplot as plt
from functools import reduce

from drg_tools.plotlib import plot_attribution




if __name__ == '__main__':
    ism1 = np.load(sys.argv[1], allow_pickle = True)
    ism2 = np.load(sys.argv[2], allow_pickle = True)
    ref = np.load(sys.argv[3], allow_pickle = True)
    
    select = sys.argv[4]
    elect = sys.argv[5]
    
    names1, values1, exp1 = ism1['names'], ism1['values'], ism1['experiments']
    names2, values2, exp2 = ism2['names'], ism2['values'], ism2['experiments']
    
    seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
    if len(np.shape(seqfeatures)) == 1:
        seqfeatures, featurenames = seqfeatures
    
    common = reduce(np.intersect1d, [names1, names2, genenames])
    exp = np.intersect1d(exp1, exp2)
    nsort1 = np.argsort(names1)[np.isin(np.sort(names1),common)]
    nsort2 = np.argsort(names2)[np.isin(np.sort(names2),common)]
    gsort = np.argsort(genenames)[np.isin(np.sort(genenames), common)]
    values1 = values1[nsort1]
    values2 = values2[nsort2]
    esort1 = np.argsort(exp1)[np.isin(np.sort(exp1), exp)]
    values1 = values1[:,esort1]
    esort2 = np.argsort(exp2)[np.isin(np.sort(exp2), exp)]
    values2 = values2[:,esort2]
    seqfeatures = seqfeatures[gsort]
    values1 = np.transpose(values1, axes = (0,1,3,2))
    values2 = np.transpose(values2, axes = (0,1,3,2))
    outname = os.path.splitext(sys.argv[1])[0]+'-vs-'+os.path.splitext(sys.argv[2])[0]+'_'+select+'_'+elect
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    if isint(select):
        select = int(select)
    else:
        select = list(common).index(select)
        
    if isint(elect):
        elect = int(elect)
    else:
        elect = list(exp).index(elect)
    
    seq = seqfeatures[select]
    att1 = values1[select, elect]
    att2 = values2[select, elect]
    print(common[select], exp[elect])
    
    mlocs = None
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile])
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in sys.argv[1]]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    seq_based = True
    if '--attributions' in sys.argv:
        seq_based = False

    att = np.array([att1, att2, att1-att2])
    
    fig = plot_attribution(seq, att, exp=['A', 'B', 'A-B'], motifs = mlocs, seq_based = seq_based)
    if '--show' in sys.argv:
        plt.show()
    else:
        fig.savefig(outname+'.jpg', dpi = 200, bbox_inches = 'tight')



