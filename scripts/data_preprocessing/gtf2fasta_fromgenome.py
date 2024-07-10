import numpy as np
import sys, os
import gzip 

def readfasta(fasta):
    fasta = gzip.open(fasta, 'rt').readlines()
    fseq = ''
    for l, line in enumerate(fasta):
        if l !=0:
            line = line.strip('\n').upper()
            fseq += line
    return fseq

def reverse_complement(seq):
    rseq = np.array(list(seq))[::-1]
    nseq = np.copy(rseq)
    nseq[rseq == 'A'] = 'T'
    nseq[rseq == 'C'] = 'G'
    nseq[rseq == 'G'] = 'C'
    nseq[rseq == 'T'] = 'A'
    return ''.join(nseq)

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
        
    

chrfold = sys.argv[1] # folder with all chromosomes in it
infofile = sys.argv[2] # gtffile that contains all the locations for the regions

outname = sys.argv[1].strip('/').split('/')[-1]+os.path.splitext(os.path.split(sys.argv[2].strip('.gz'))[1])[0]
print(outname)

# bedfile order:
#chr start end name number_exon strand
ifile = readgtf(infofile)
print(np.shape(ifile))


if '--generate_transcripts' in sys.argv:
    mask =ifile[:,4] == 'exon'
    ifile = ifile[mask]
    outname += '.trscrpt'

elif '--from_tss' in sys.argv:
    mask =ifile[:,4] == 'gene'
    ifile = ifile[mask]
    outname += '.tss'

elif '--seqtype' in sys.argv:
    seqtype = sys.argv[sys.argv.index('--seqtype')+1]
    mask =ifile[:,4] == seqtype
    ifile = ifile[mask]
    outname += '.'+seqtype
print(np.shape(ifile))

if '--genetype' in sys.argv:
    genetype = sys.argv[sys.argv.index('--genetype')+1]
    mask =ifile[:,6] == genetype
    ifile = ifile[mask]
    outname += '.'+genetype
print(np.shape(ifile))

if '--usegeneid' in sys.argv:
    names = ifile[:,-3]
elif '--usegeneid_noversion':
    vnames = ifile[:,-3]
    names = []
    for v, vn in enumerate(vnames):
        if '.' in vn:
            vn = vn.split('.')[0]
        names.append(vn)
    names = np.array(names)
else:
    names = ifile[:,-1]


offset = 0
if '--oneindex' in sys.argv:
    # USE this if the bed file counts from 1-X and not from 0 to X
    offset = 1

flank = 0
if '--add_flanks' in sys.argv:
    flank = int(sys.argv[sys.argv.index('--add_flanks')+1])
    outname += 'flank'+str(flank)

if '--list' in sys.argv:
    thelist = np.genfromtxt(sys.argv[sys.argv.index('--list')+1], dtype = str)
    mask = np.isin(names, thelist)
    names, ifile = names[mask], ifile[mask]
    print(len(names), 'left')

# iterate over all chromosomes in the infofile
uchroms = np.unique(ifile[:,0])
print(uchroms)

print(outname)
# generate fasta file with sequences
outfasta = open(outname+'.fasta', 'w')
for u, uchr in enumerate(uchroms):
    if os.path.isfile(chrfold.rstrip('/')+'/'+uchr+'.fa.gz'):
        print('Read',chrfold.rstrip('/')+'/'+uchr+'.fa.gz') 
        chrfasta = readfasta(chrfold.rstrip('/')+'/'+uchr+'.fa.gz')
        print(len(chrfasta))
        chrextra = 'N' * flank
        chrfasta = chrextra + chrfasta + chrextra
        print(len(chrfasta))
        chrmask = ifile[:,0] == uchr
        print('Generate seq for '+uchr)
        unames, indx = np.unique(names[chrmask], return_index = True)
        unames = unames[np.argsort(indx)]
        print('Genes in chr', len(unames))
        for n, na in enumerate(unames):
            chrmask = np.where(names == na)[0]
            
            if '--generate_transcripts' in sys.argv:
                extseq = ''
                for e in chrmask:
                    ext= chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                    if ifile[e, 3] == '-':
                        ext = reverse_complement(ext)
                        extseq = ext + extseq
                    else:
                        extseq = extseq + ext
                outfasta.write('>'+na+'\n'+extseq+'\n')
            elif '--from_tss' in sys.argv:
                e = chrmask[0]
                
                if ifile[e, 3] == '-':
                    extseq = chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                    extseq = reverse_complement(extseq)
                    outfasta.write('>'+na+'\n'+extseq+'\n')
                elif ifile[e, 3] == '+':
                    extseq = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                    outfasta.write('>'+na+'\n'+extseq+'\n')
                elif ifile[e, 3] == '.':
                    extseqf = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                    extseqb= chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                    extseqb = reverse_complement(extseq)
                    outfasta.write('>'+na+'\n'+extseqf+'\n'+'>'+na+'.rev\n'+extseqb+'\n')
            else:
                e = chrmask[0]
                extseq = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                if ifile[e, 3] == '-':
                        extseq = reverse_complement(extseq)
                outfasta.write('>'+na+'\n'+extseq+'\n')
            
            
        


    
