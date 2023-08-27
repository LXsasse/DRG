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


chrfold = sys.argv[1] # folder with all chromosomes in it
infofile = sys.argv[2] # bedfile that contains all the locations for the regions
outname = sys.argv[1].strip('/').split('/')[-1]+os.path.splitext(os.path.split(sys.argv[2])[1])[0]


# bedfile order:
#chr start end name number_exon strand
ifile = np.genfromtxt(infofile, dtype = str)
has_strand = np.shape(ifile)[1] >= 5

# iterate over all chromosomes in the infofile
uchroms = np.unique(ifile[:,0])

offset = 0
if '--oneindex' in sys.argv:
    # USE this if the bed file counts from 1-X and not from 0 to X
    offset = 1

flank = 0
if '--add_flanks' in sys.argv:
    flank = int(sys.argv[sys.argv.index('--add_flanks')+1])
    outname += 'flank'+str(flank)

if '--generate_transcripts' in sys.argv:
    outname += '.trscrpt'
elif '--from_tss' in sys.argv:
    outname += '.tss'


print(outname)
# generate fasta file with sequences
outfasta = open(outname+'.fasta', 'w')
for u, uchr in enumerate(uchroms):
    if os.path.isfile(chrfold.rstrip('/')+'/'+uchr+'.fa.gz'):
        print('Read',chrfold.rstrip('/')+'/'+uchr+'.fa.gz') 
        chrfasta = readfasta(chrfold.rstrip('/')+'/'+uchr+'.fa.gz')
        chrmask = ifile[:,0] == uchr
        print('Generate seq for '+uchr)
        unames, indx = np.unique(ifile[chrmask,3], return_index = True)
        unames = unames[np.argsort(indx)]
        print('Genes in chr', len(unames))
        for n, na in enumerate(unames):
            chrmask = np.where(ifile[:,3] == na)[0]
            if '--generate_transcripts' in sys.argv:
                extseq = ''
                for e in chrmask:
                    ext= chrfasta[int(ifile[e, 1])-offset-flank:int(ifile[e,2])-offset+flank]
                    if ifile[e, 5] == '-':
                        ext = reverse_complement(ext)
                    extseq += ext
                outfasta.write('>'+ifile[e,3]+'\n'+extseq+'\n')
            elif '--from_tss' in sys.argv:
                e = chrmask[0]
                
                if ifile[e, 5] == '-':
                    extseq = chrfasta[int(ifile[e, 2])-offset-flank:int(ifile[e,2])-offset+flank]
                    extseq = reverse_complement(extseq)
                    outfasta.write('>'+ifile[e,3]+'\n'+extseq+'\n')
                elif ifile[e, 5] == '+':
                    extseq = chrfasta[int(ifile[e, 1])-offset-flank:int(ifile[e,1])-offset+flank]
                    outfasta.write('>'+ifile[e,3]+'\n'+extseq+'\n')
                elif ifile[e, 5] == '.':
                    extseqf = chrfasta[int(ifile[e, 1])-offset-flank:int(ifile[e,1])-offset+flank]
                    extseqb= chrfasta[int(ifile[e, 2])-offset-flank:int(ifile[e,2])-offset+flank]
                    extseqb = reverse_complement(extseq)
                    outfasta.write('>'+ifile[e,3]+'\n'+extseqf+'\n'+'>'+ifile[e,3]+'.rev\n'+extseqb+'\n')
            else:
                e = chrmask[0]
                extseq = chrfasta[int(ifile[e, 1])-offset-flank:int(ifile[e,2])-offset+flank]
                if has_strand:
                    if ifile[e, 5] == '-':
                        extseq = reverse_complement(extseq)
                outfasta.write('>'+ifile[e,3]+'\n'+extseq+'\n')
            
            
        


    
