import numpy as np
import sys, os
import matplotlib.pyplot as plt

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

def plotHist(x, y = None, xcolor='navy', yaxis = False, xalpha= 0.5, ycolor = 'indigo', yalpha = 0.5, addcumulative = False, bins = None, xlabel = None, title = None, logx = False, logy = False, logdata = False):
    fig = plt.figure(figsize = (3.5,3.5))
    axp = fig.add_subplot(111)
    axp.spines['top'].set_visible(False)
    axp.spines['right'].set_visible(False)
    
    if logdata:
        x = np.log10(x+1)
    
    a,b,c = axp.hist(x, bins = bins, color = xcolor, alpha = xalpha)
    print(b)
    print(a)
    if y is not None:
        ay,by,cy = axp.hist(y, bins = bins, color = ycolor, alpha = yalpha)
        print(ay)
    
    if addcumulative != False:
        axp2 = axp.twinx()
        axp2.spines['top'].set_visible(False)
        axp2.spines['left'].set_visible(False)
        axp2.tick_params(bottom = False, labelbottom = False)
        axp2.set_yticks([0.25,0.5,0.75,1])
        axp2.set_yticklabels([25,50,75,100])
        if addcumulative == 2:
            addcumulative = 1
            ag_,bg_,cg_ = axp2.hist(x, color = 'maroon', alpha = 1, density = True, bins = bins, cumulative = -1, histtype = 'step')
        ag,bg,cg = axp2.hist(x, color = xcolor, alpha = 1, density = True, bins = bins, cumulative = addcumulative, histtype = 'step')
        if y is not None:
            agy,bgy,cgy = axp2.hist(y, color = ycolor, alpha = 1, density = True, bins = bins, cumulative = addcumulative, histtype = 'step')
    
    
    
    if yaxis:
        print('yaxis',np.amax(a))
        axp.plot([0,0], [0, np.amax(a)], c = 'k', zorder = 5)
    
    if logx:
        if addcumulative:
            axp2.set_xscale('symlog')
        axp.set_xscale('symlog')
        
    if logy:
        axp.set_yscale('symlog')
    
    if xlabel is not None:
        axp.set_xlabel(xlabel)
    if title is not None:
        axp.set_title(title)
    return fig

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

    fig = plotHist(data, y = altdata, bins = bins, xlabel = xlabel, addcumulative = addcumul, yaxis = ploty, logx = logx, logy = logy, logdata = logdata)



    fig.savefig(outname + '.jpg', dpi = 300, bbox_inches='tight')
    print(outname+'.jpg')
