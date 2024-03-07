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


def logoax(fig, att, ylabel = None, ylim = None, sb = 111, pos = None, labelbottom = True, bottom = True):
    ax0 =  fig.add_subplot(sb[0], sb[1], sb[2])
    if pos is not None:
        ax0.set_position(pos)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.tick_params(bottom = bottom, labelbottom = labelbottom)
    att = pd.DataFrame({'A':att[:,0],'C':att[:,1], 'G':att[:,2], 'T':att[:,3]})
    lm.Logo(att, ax = ax0)
    if ylabel is not None:
        ax0.set_ylabel(ylabel)
    if ylim is not None:
        ax0.set_ylim(ylim)
    return ax0
    
def heatax(ism, fig, pos = None, sb = 111, cmap = 'coolwarm', ylabel = None, labelbottom = True, bottom = True, vlim = None):
    if vlim is None:
        vlim = np.amax(np.absolute(ism))
    ax1 =fig.add_subplot(sb)
    if pos is not None:
        ax1.set_position(pos)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.imshow(ism.T, aspect = 'auto', cmap = cmap, vmin = -vlim, vmax = vlim)
    if ylabel is not None:
        ax1.set_ylabel(ylabel)
    ax1.tick_params(bottom = bottom, labelbottom = labelbottom)
    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(['A','C','G','T'])
    return ax1

def plot_attribution(seq, att, motifs = None, seq_based = 1, exp = None, vlim = None, unit = 0.15, ratio = 10, ylabel = None):
    #print(att[0,:10,:,0], att[0,:10,:,0])
    ism = np.copy(att)
    if seq_based:
        att = seq * att
        ylabel = 'Attribution\nat ref'
    
    if ylabel is None:
        ylabel = 'Attribution'
    
    if exp is None:
        exp = np.arange(len(att), dtype = int).astype(str)
        
    if vlim is None:
        mina = min(0,np.amin(np.sum(np.ma.masked_greater(att,0), axis = -1)))
        maxa = np.amax(np.sum(np.ma.masked_less(att,0), axis = -1))
        attlim = [mina, maxa]
    else:
        attlim = vlim
    
    fig = plt.figure(figsize = (unit*len(seq), len(att) * ratio*unit), dpi = 50)
    
    axs = []
    for a, at in enumerate(att):
        axs.append(logoax(fig, at, ylabel = exp[a], ylim = attlim, sb = [len(att), 1, 1+a], pos = [0.1,0.1+(len(att)-1-a)/len(att)*0.8,0.8,0.8*(1/len(att))*0.8], labelbottom = a == len(att)-1, bottom = a == len(att)-1))
    
    if motifs is not None:
        mask = motifs[:,-2] == 0
        colors = motifs[mask,-1]
        #print(motifs[mask,1])
        locations = [ti1[l] for l in motifs[mask,1]]
        #print(locations)
        add_frames(att, locations, colors, ax0)

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
    
    select = sys.argv[3]
    elect = sys.argv[4]
    
    names, values, exp = ism['names'], ism['values'], ism['experiments']
    print(exp)
    seqfeatures, genenames = ref['seqfeatures'], ref['genenames']
    if len(np.shape(seqfeatures)) == 1:
        seqfeatures, featurenames = seqfeatures
    
    nsort = np.argsort(names)[np.isin(np.sort(names), genenames)]
    gsort = np.argsort(genenames)[np.isin(np.sort(genenames), names)]
    values = values[nsort]
    seqfeatures = seqfeatures[gsort]
    names, genenames = names[nsort], genenames[gsort]
    values = np.transpose(values, axes = (0,1,3,2))
    outname = os.path.splitext(sys.argv[1])[0] +'_'+select+'_'+elect
    
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    
    if isint(select):
        select = int(select)
    else:
        select = list(names).index(select)
    
    if elect == 'all':
        elect = np.arange(len(exp), dtype = int)
    else:
        elect = elect.split(',')
        for e, el in enumerate(elect):
            if isint(el):
                elect[e] = int(el)
            else:
                elect[e] = list(exp).index(el)
    
    seq = seqfeatures[select]
    att = values[select][elect]
    exp = exp[elect]
    print(names[select], genenames[select], exp)
    
    mlocs = None
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile])
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in sys.argv[1]]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    
    if '--centerattributions' in sys.argv:
        att -= (np.sum(att, axis = -1)/4)[...,None]
    elif '--decenterattributions' in sys.argv:
        att -= seq * att
    elif '--meaneffectattributions' in sys.argv:
        att -= (np.sum((seq == 0)*att, axis = -1)/3)[...,None]
    
    seq_based = True
    if '--showall_attributions' in sys.argv:
        seq_based = False

    vlim = None
    if '--vlim' in sys.argv:
        vlim = np.array(sys.argv[sys.argv.index('--vlim')+1].split(','), dtype = float)
    
    unit = 0.15
    if '--unit' in sys.argv:
        unit = float(sys.argv[sys.argv.index('--unit')+1])
    ratio = 10
    if '--ratio' in sys.argv:
        ratio = float(sys.argv[sys.argv.index('--ratio')+1])
    
    if '--center' in sys.argv:
        flanks = int(sys.argv[sys.argv.index('--center')+1])
        st = int((len(seq)-2*flanks)/2)
        seq = seq[st:-st]
        att = att[:,st:-st]
     
    if '--locations' in sys.argv:
        loc = sys.argv[sys.argv.index('--locations')+1].split(',')
        seq = seq[int(loc[0]):int(loc[1])]
        att = att[:,int(loc[0]):int(loc[1])]
    
    dpi = 200
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
    
    fig = plot_attribution(seq, att, motifs = mlocs, seq_based = seq_based, exp = exp, vlim = vlim, unit = unit, ratio = ratio)
    if '--show' in sys.argv:
        plt.show()
    else:
        if '--transparent' in sys.argv:
            fig.savefig(outname+'.png', transparent = True, dpi = dpi, bbox_inches = 'tight')
        else:
            fig.savefig(outname+'.jpg', dpi = dpi, bbox_inches = 'tight')
            print(outname+'.jpg')



# algign sequences and plot attributions of both sequences

