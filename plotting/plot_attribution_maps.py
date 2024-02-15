import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd


def add_frames(att, locations, colors, ax):
    att = np.array(att)
    cmap = ['purple', 'limegreen']
    for l, loc in enumerate(locations):
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = cmap[colors[l]])
        ax.plot(x, [maxa, maxa], c = cmap[colors[l]])
        ax.plot([x[0], x[0]] , [mina, maxa], c = cmap[colors[l]])
        ax.plot([x[1], x[1]] , [mina, maxa], c = cmap[colors[l]])


def plot_attribution(seq, att, motifs = None, seq_based = True, figscale = 0.15, ylim = None):
    #print(att[0,:10,:,0], att[0,:10,:,0])
    ism = np.copy(att)
    if seq_based:
        att = -np.sum(att, axis = -1)/3
        att = seq * att[:, None]
        ylabel = 'ISM\nmean'
    else:
        att -= (np.sum(att, axis = -1)/3)[..., None]
        ylabel = 'centered ISM'
    
    
    mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -1)))
    maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -1))
    attlim = [mina, maxa]

    fig = plt.figure(figsize = (figscale*len(seq), 10*figscale+figscale*5), dpi = 50)
    
    ax0 =  fig.add_subplot(211)
    ax0.set_position([0.1,0.1+(5/15)*0.8,0.8,0.8*(10/15)])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.tick_params(bottom = False, labelbottom = False)
    att = pd.DataFrame({'A':att[:,0],'C':att[:,1], 'G':att[:,2], 'T':att[:,3]})
    lm.Logo(att, ax = ax0)
    ax0.set_ylabel(ylabel)
    if ylim is not None:
        ax0.set_ylim(ylim)
    else:
        ax0.set_ylim(attlim)

    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        #print(motifs[mask,1])
        locations = [ti1[l] for l in motifs[mask,1]]
        #print(locations)
        add_frames(att, locations, colors, ax0)

    vlim = np.amax(np.absolute(ism))
    ax1 =fig.add_subplot(212)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ta_ = ax1.imshow(ism.T, aspect = 'auto', cmap = 'coolwarm', vmin = -vlim, vmax = vlim)
    ax1.set_ylabel('ISM')
    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(['A','C','G','T'])
    ax1.set_position([0.1,0.1,0.8,0.8*(4/15)])
    
    axc =fig.add_subplot(991)
    #axc.spines['top'].set_visible(False)
    #axc.spines['right'].set_visible(False)
    axc.imshow(np.linspace(0,1,101).reshape(-1,1), aspect = 'auto', cmap = 'coolwarm', vmin = 0, vmax = 1)
    axc.set_position([0.9+0.25/len(seq),0.1,1/len(seq),0.8*(4/15)])
    axc.set_yticks([0,100])
    axc.set_yticklabels([-round(vlim,2), round(vlim,2)])
    axc.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False, labelright = True, right = True)
    
    return fig


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

    seq_based = True
    if '--attributions' in sys.argv:
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
        
    fig = plot_attribution(seq, att, motifs = mlocs, seq_based = seq_based, ylim = ylim)
    if '--show' in sys.argv:
        plt.show()
    else:
        fig.savefig(outname+'.jpg', dpi = 200, bbox_inches = 'tight')



# algign sequences and plot attributions of both sequences

