
'''
Plots ATAC counts as bar plots against each other in different colors
'''

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from drg_tools.sequence_utils import avgpool
from drg_tools.io_utils import insint, readinfasta


if __name__ == '__main__':
    bpcounts = np.load(sys.argv[1], allow_pickle = True)
    names = bpcounts['names']

    if 'celltypes' in bpcounts.files:
        celltypes = bpcounts['celltypes']
    else:
        celltypes = None
        

    if celltypes is not None:
        print('celltypes')
        for c, cellt in enumerate(celltypes):
            print(c, cellt)

    counts = bpcounts['counts']
    print(np.shape(counts))
    outname = os.path.splitext(os.path.split(sys.argv[1])[1])[0]


    if '--combine_tracks' in sys.argv:
        tracknames = sys.argv[sys.argv.index('--combine_tracks')+1].split(',')
        if isint(tracknames[0]):
            track = np.array(tracknames, dtype = int)
        elif celltypes is not None:
            track = np.where(np.isin(celltypes, tracknames))[0]
        else:
            print('tracks could not be determined')
            sys.exit()
        counts = counts[:,track]
        outname += 't'+'-'.join(track.astype(str))
    elif '--track' in sys.argv:
        tracknames = sys.argv[sys.argv.index('--track')+1]
        if isint(tracknames):
            track = [int(tracknames)]
        elif celltypes is not None:
            track = np.where(np.isin(celltypes, tracknames))[0]
        else:
            print('track could not be determined')
            sys.exit()
        counts = counts[:,track]
        outname += 't'+'-'.join(track.astype(str))
    else:
        track = np.arange(np.shape(counts)[1], dtype = int)

    print(track, np.shape(counts))
    counts = np.sum(counts,axis = 1)
    print(np.shape(counts))

    if '--cut_edges' in sys.argv:
        cedges = int(sys.argv[sys.argv.index('--cut_edges')+1])
        counts = counts[:,cedges:-cedges]
        outname += 'cut'+str(cedges)

    if '--pool' in sys.argv:
        npool = int(sys.argv[sys.argv.index('--pool')+1])
        counts = avgpool(counts, npool)
        
    if '--norm' in sys.argv or '--normmse' in sys.argv:
        counts -= np.amin(counts, axis = -1)[:,None]
        countn = np.sum(counts, axis = -1)[:,None]
        countn[countn == 0] = 1
        counts = counts.astype(np.float32)
        counts /= countn
        counts *= 100

        


    pcounts = None
    if '--predicted_counts' in sys.argv:
        bpcounts = np.load(sys.argv[sys.argv.index('--predicted_counts')+1], allow_pickle = True)
        outname += '-vs-'+os.path.splitext(os.path.split(sys.argv[sys.argv.index('--predicted_counts')+1])[1])[0]
        pnames = bpcounts['names']
        pcounts = bpcounts['counts']
        pcelltypes = None
        if 'celltypes' in bpcounts.files:
            pcelltypes = bpcounts['celltypes']
        psort = np.argsort(pnames)[np.isin(np.sort(pnames), names)]
        sort = np.argsort(names)[np.isin(np.sort(names), pnames)]
        names, counts = names[sort], counts[sort]
        pnames, pcounts = pnames[psort], pcounts[psort]
        if '--combine_tracks' in sys.argv:
            if pcelltypes is not None:
                track = np.where(np.isin(celltypes, tracknames))[0]
            elif np.shape(pcounts)[1] > 1:
                print('tracks could not be determined')
                sys.exit()
            else:
                track = [0]
            pcounts = np.sum(pcounts[:,track], axis = 1)
                
        elif '--track' in sys.argv:
            if pcelltypes is not None:
                track = np.where(np.isin(celltypes, tracknames))[0]
            elif np.shape(pcounts)[1] > 1:
                print('track could not be determined in', pcelltypes)
                sys.exit()
            else:
                track = [0]
            pcounts = np.sum(pcounts[:,track], axis = 1)
        else:
            pcounts = np.sum(pcounts, axis = 1)
        if '--cut_edges' in sys.argv:
            pcounts = pcounts[:,cedges:-cedges]
        if '--pool' in sys.argv:
            pcounts = avgpool(pcounts, npool)
        if '--norm' in sys.argv or '--normcounts' in sys.argv:
            pcounts -= np.amin(pcounts, axis = -1)[:,None]
            pcountn = np.sum(pcounts, axis = -1)[:,None]
            pcountn[pcountn == 0] = 1
            pcounts = pcounts.astype(np.float32)
            pcounts /= pcountn
            if '--normcounts' in sys.argv:
                pcounts *= np.sum(counts,axis = 1)[:,None]
            else:
                pcounts *= 100
        elif '--normmse' in sys.argv:
            for c, cou in enumerate(counts):
                scale = nnls(pcounts[[c]].T, cou)
                pcounts[c] *= scale[0]
                


    if '--predicted_counts' in sys.argv and '--sortbydif' in sys.argv:
        countper = np.sum(np.absolute(counts-pcounts), axis = 1)
        sort = np.argsort(countper)
        
    elif '--dontsort' in sys.argv:
        sort = np.arange(len(counts), dtype = int)
    else:
        countper = np.sum(counts, axis = 1)
        sort = np.argsort(-countper)


    names, counts = names[sort], counts[sort]
    if '--predicted_counts' in sys.argv:
        pcounts = pcounts[sort]
    

    plotnames = names
    if '--list_of_regions' in sys.argv:
        regions = np.genfromtxt(sys.argv[sys.argv.index('--list_of_regions')+1], dtype = str)
        mask = [list(names).index(i) for i in regions[:, 0] if i in names]
        plotnames = [regions[i][0] + ' (-log10p:'+regions[i][1]+')' for i in range(len(regions)) if regions[i][0] in names]
        counts = counts[mask]
        names = names[mask]
        if pcounts is not None:
            pcounts = pcounts[mask]

    sequences = None
    if '--sequences' in sys.argv:
        sfile = sys.argv[sys.argv.index('--sequences')+1]
        snames, seqs = readinfasta(sfile)
        sort = np.argsort(snames)[np.isin(np.sort(snames), names)]
        sequences = seqs[sort]
        
        
        
    for n, name in enumerate(plotnames):
        fig = plt.figure(figsize = (max(0.05*np.shape(counts)[-1],5.5),3.5), dpi = 200)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.bar(np.arange(np.shape(counts)[-1]), counts[n], label = 'Total: '+str(int(np.sum(counts[n]))), alpha = 0.9, color = 'blue')
        #ax.plot(np.arange(np.shape(counts)[-1]), counts[n])
        if pcounts is not None:
            ax.bar(np.arange(np.shape(counts)[-1]), pcounts[n], label = 'Predicted: '+str(int(np.sum(pcounts[n]))), alpha = 0.9, color= 'red')
            combined_count = np.amin(np.array([pcounts[n], counts[n]]),axis = 0)
            ax.bar(np.arange(len(combined_count)), combined_count, label = 'Overlapping: '+str(int(np.sum(combined_count))), alpha = 0.9, color= 'dimgrey')
            #ax.plot(np.arange(np.shape(counts)[-1]), pcounts[n])
        if sequences is not None:
            ax.set_xticks(np.arange(np.shape(counts)[-1]))
            ax.set_xticklabels(list(sequences[n]), fontsize = 3.5)
            ax.tick_params(bottom = False)
        ax.legend()
        ax.set_title(name)
        
        if '--plot_difference' in sys.argv and pcounts is not None:
            figd = plt.figure(figsize = (0.05*np.shape(counts)[-1],3.5), dpi = 200)
            axd = figd.add_subplot(111)
            difpos, difneg = counts[n]-pcounts[n], counts[n]-pcounts[n]
            difpos[difpos < 0] = 0 
            difneg[difneg > 0] = 0
            axd.bar(np.arange(np.shape(counts)[-1]), difpos, label = 'Difpos: '+str(int(np.sum(difpos))), alpha = 0.9, color = 'blue')
            axd.bar(np.arange(np.shape(counts)[-1]), difneg, label = 'Difneg: '+str(int(np.sum(difneg))), alpha = 0.9, color = 'red')
            #axd.plot(np.arange(np.shape(counts)[-1]), counts[n]-pcounts[n])
            axd.legend()
            axd.set_title('Delta '+name)
            
        
        if '--savefig' in sys.argv:
            print(outname+names[n]+'.jpg')
            fig.savefig(outname+names[n]+'.jpg', dpi = 150, bbox_inches = 'tight')
            if '--plot_difference' in sys.argv:
                figd.savefig(outname+names[n]+'_dif.jpg', dpi = 150, bbox_inches = 'tight')
        plt.show()
        plt.close()
        
        
        
