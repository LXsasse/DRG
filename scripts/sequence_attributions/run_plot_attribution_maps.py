import numpy as np
import sys, os
import matplotlib.pyplot as plt

from drg_tools.plotlib import plot_attribution


def isint(x):
    try:
        int(x) 
        return True
    except:
        return False


if __name__ == '__main__':
    ism = np.load(sys.argv[1], allow_pickle = True)
    ref = np.load(sys.argv[2], allow_pickle = True)
    ismfile = sys.argv[1].rsplit('.',1)[-1]
    reffile = sys.argv[2].rsplit('.',1)[-1]
    
    outname = os.path.splitext(sys.argv[1])[0]
    select=None
    elect = None
    
    if ismfile == 'npz':
        names, values, exp = ism['names'], ism['values'], ism['experiments']
        values = np.transpose(values, axes = (0,1,3,2))
    else:
        values = ism
    if reffile == 'npz':
        seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
        if len(np.shape(seqfeatures)) == 1:
            seqfeatures, featurenames = seqfeatures
    else:
        seqfeatures = ref
    
    if len(np.shape(seqfeatures)) > 2:
        select = sys.argv[3]
        if isint(select):
            select = int(select)
        else:
            select = list(genenames).index(select)
        seq = seqfeatures[select]
    else:
        seq = seqfeatures
        
    if len(np.shape(values)) > 2:
        select = sys.argv[3]
        if isint(select):
            select = int(select)
        else:
            select = list(names).index(select)
        
        att = values[select]
        
        if len(np.shape(att))>2:
            elect = sys.argv[4]
            if isint(elect):
                elect = int(elect)
            else:
                elect = list(exp).index(elect)
            att = att[elect]
    else:
        att = ism 


    if select is not None:
         outname += str(select)
    if elect is not None:
         outname += str(elect)
    
    mlocs = None
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile])
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in sys.argv[1]]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    centatt = False
    if '--centerattributions' in sys.argv:
        att -= (np.sum(att, axis = -1)/4)[...,None]
    elif '--decenterattributions' in sys.argv:
        att -= seq * att
    elif '--meaneffectattributions' in sys.argv:
        att -= (np.sum((seq == 0)*att, axis = -1)/3)[...,None]
    elif '--centerattributionsforvisual' in sys.argv:
        centatt = True
    
    
    seq_based = True
    if '--showall_attributions' in sys.argv:
        seq_based = False

    if '--transpose_seq' in sys.argv:
        seq = seq.T
        print(np.shape(att))
    if '--transpose_ism' in sys.argv:
        att = att.T
        print(np.shape(seq))
        
    if '--center' in sys.argv:
        w = int(sys.argv[sys.argv.index('--center')+1])
        att = att[w:-w]
        seq = seq[w:-w]
      
    ylim = None
    if '--ylim' in sys.argv:
        ylim = sys.argv[sys.argv.index('--ylim')+1].split(',')
        ylim = np.array(ylim, dtype = float)
        print(ylim)
        
    fig = plot_attribution(seq, att, motifs = mlocs, seq_based = seq_based, center_attribution = centatt, ylim = ylim)
    if '--show' in sys.argv:
        plt.show()
    else:
        fig.savefig(outname+'.jpg', dpi = 200, bbox_inches = 'tight')



# algign sequences and plot attributions of both sequences

