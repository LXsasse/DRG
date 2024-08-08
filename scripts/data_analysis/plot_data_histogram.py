import numpy as np
import sys, os
import matplotlib.pyplot as plt

from drg_tools.plotlib import plotHist

def read(f, column = 1):
    lines = open(f, 'r').readlines()
    names = []
    data = []
    for l, line in enumerate(lines):
        if line[0] != '#':
            line = line.strip().split()
            names.append(line[0])
            data.append(line[column])
    names = np.array(names)
    data = np.array(data, dtype = float)
    return names, data


if __name__ == '__main__':

    column = -1
    if '--column' in sys.argv:
        column = int(sys.argv[sys.argv.index('--column')+1])
        
    names, data = read(sys.argv[1], column = column)
    outname = os.path.splitext(sys.argv[1])[0]+'_hist'+str(column)

    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]

    

    altdata = None
    if '--compareto' in sys.argv:
        compare = sys.argv[sys.argv.index('--compareto')+1]
        column = int(sys.argv[sys.argv.index('--compareto')+2])
        outname += '_vs_'+os.path.splitext(compare)[0]+'_h'+str(column)
        altnames, altdata = read(compare, column = column)
        #sort = np.argsort(altnames)[np.isin(np.sort(altnames), names)]
        #altdata = altdata[sort]

    elif '--devideintoclasses' in sys.argv:
        classes = sys.argv[sys.argv.index('--devideintoclasses')+1] 
        altnames, altdata = read(classes, column = -1)
        sortalt, sort = np.argsort(altnames)[np.isin(np.sort(altnames), names)], np.argsort(names)[np.isin(np.sort(names), altnames)]
        altnames, altdata = altnames[sortalt], altdata[sortalt]
        names, data = names[sort], data[sort]
        altnames, altdata = names[altdata == 1], data[altdata == 1]

    
    if '--count_classes' in sys.argv:
        ud, data = np.unique(data, return_counts = True)
        if '--remove_class' in sys.argv:
            rclass = sys.argv[sys.argv.index('--remove_class')+1]
            data = data[ud != float(rclass)]


    xlabel = None
    if '--xlabel' in sys.argv:
        xlabel = sys.argv[sys.argv.index('--xlabel')+1]

    bins = None
    if '--bins' in sys.argv:
        bins = sys.argv[sys.argv.index('--bins')+1]
        if ',' in bins:
            bins = np.array(bins.split(','),dtype=float)
            bins = np.linspace(bins[0], bins[1], int(bins[2]))
            print(bins)
        else:
            bins = int(bins)

    ploty = False
    if '--plotyaxis' in sys.argv:
        ploty = True

    addcumul = False
    if '--addcumulative' in sys.argv:
        addcumul = int(sys.argv[sys.argv.index('--addcumulative')+1])

    if '--xlim' in sys.argv:
        xlim = np.array(sys.argv[sys.argv.index('--xlim')+1].split(','))

    logx = False
    if '--logx' in sys.argv:
        logx = True

    logy = False
    if '--logy' in sys.argv:
        logy = True

    logdata = False
    if '--logdata' in sys.argv:
        logdata = True

    fig = plotHist(data, y = altdata, bins = bins, xlabel = xlabel, addcumulative = addcumul, add_yaxis = ploty, logx = logx, logy = logy, logdata = logdata)



    fig.savefig(outname + '.jpg', dpi = 300, bbox_inches='tight')
    print(outname+'.jpg')
