import numpy as np
import sys, os
import gzip

def readgtf(g):
    if os.path.splitext(g)[1] == '.gz':
        obj = gzip.open(g, 'rt')
    else:
        obj = open(g,'r')
    fi = obj.readlines()
    itype = []
    start, end = [], []
    chrom = []
    strand = []
    gene_id, gene_type, gene_name = [], [], []
    
    for l, line in enumerate(fi):
        if line[0] != '#':
            line = line.strip().split('\t')
            chrom.append(line[0])
            itype.append(line[2])
            start.append(int(line[3]))
            end.append(int(line[4]))
            strand.append(line[6])
            info = line[8].split(';')
            gid, gty, gna = '' ,'', ''
            for i, inf in enumerate(info):
                inf = inf.strip()
                if inf[:7] == 'gene_id':
                    inf = inf.split()
                    gid = inf[1].strip('"')
                if inf[:9] == 'gene_type':
                    inf = inf.split()
                    gty = inf[1].strip('"')
                if inf[:9] == 'gene_name':
                    inf = inf.split()
                    gna = inf[1].strip('"')
            gene_id.append(gid)
            gene_name.append(gna)
            gene_type.append(gty)
    return np.array([chrom, start, end, strand, itype, gene_id, gene_type, gene_name]).T


def groupings(tlen, groupsizes, kfold):
    groups = []
    csize = []
    avail = np.arange(len(groupsizes), dtype = int)
    while True:
        if len(avail) < 1 or len(csize) == kfold:
            break
        seed = np.random.choice(avail)
        group = np.array([seed])
        avail = avail[~np.isin(avail, group)]
        gdist = abs(tlen-np.sum(groupsizes[group]))
        while True:
            if len(avail) < 1:
                groups.append(group)
                csize.append(int(np.sum(groupsizes[group])))
                break
            ngr = avail.reshape(-1,1)
            egr = np.repeat(group, len(ngr)).reshape(len(group), len(ngr)).T
            pgr = np.append(egr, ngr, axis = 1)
            #print(gdist, len(avail), len(group))
            pdist = np.abs(tlen-np.sum(groupsizes[pgr],axis = 1))
            if (pdist < gdist).any():
                mgr = np.argmin(pdist)
                group = pgr[mgr]
                gdist = pdist[mgr]
                avail = avail[~np.isin(avail, group)]
            else:
                groups.append(group)
                csize.append(int(np.sum(groupsizes[group])))
                break
    #print(groups, csize, avail)
    #sys.exit()
    return groups, np.array(csize), np.mean(np.abs(np.array(csize) - tlen))


def generatetesttrain(names, groups, outname, kfold = 10):
    ugroups, ugroupsize = np.unique(groups, return_counts = True)
    #print(ugroups, ugroupsize)
    n = len(names)
    st = int(n/kfold)
    cdist = st
    for i in range(10000): #sampel 10,000 random possibile combinations
        cgroups, cgroupsizes, msize = groupings(st, ugroupsize, kfold)
        #print(cgroups, cgroupsizes, msize)
        if msize < cdist:
            combgroups = cgroups
            combsize = cgroupsizes
            cdist = np.copy(msize)
    print('Best split', cdist)

    obj=open(outname, 'w')
    for j, grp in enumerate(combgroups):
        print(j, ugroups[grp], np.sum(ugroupsize[grp]) - st)
        test = names[np.isin(groups, ugroups[grp])]
        obj.write('# Set_'+str(j)+'\n' + ' '.join(test)+'\n')


chrfile = sys.argv[1]
if os.path.splitext(chrfile)[1] == '.bed': 
    bed = np.genfromtxt(chrfile, dtype = str)
    names = bed[:,3]
else:
    bed = readgtf(chrfile)
    
    mask =bed[:,4] == 'gene'
    bed = bed[mask]
    
    if '--genetype' in sys.argv:
        genetype = sys.argv[sys.argv.index('--genetype')+1]
        mask =bed[:,6] == genetype
        bed = bed[mask]
        outname += '.'+genetype
    print(np.shape(bed))
    
    if '--usegeneid' in sys.argv:
        names = bed[:,-3]
    elif '--usegeneid_noversion':
        vnames = bed[:,-3]
        names = []
        for v, vn in enumerate(vnames):
            if '.' in vn:
                vn = vn.split('.')[0]
            names.append(vn)
        names = np.array(names)
    else:
        names = bed[:,-1]
    
outname = os.path.splitext(sys.argv[1].replace('.gz', ''))[0]+'_tset10.txt'

if '--exclude' in sys.argv:
    exclude = sys.argv[sys.argv.index('--exclude')+1]
    if ',' in exclude:
        exclude = exclude.split(',')
    else:
        exclude = [exclude]
    mask = ~np.isin(bed[:,0], exclude)
    bed = bed[mask]
    names = names[mask]
        

generatetesttrain(names, bed[:,0], outname)


