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

chrfold = sys.argv[1] # folder with all chromosomes in it
infofile = sys.argv[2] # bedfile that contains all the locations for the regions
outname = sys.argv[1].strip('/').split('/')[-1]+os.path.splitext(os.path.split(sys.argv[2])[1])[0]
print(outname)
ifile = np.genfromtxt(infofile, dtype = str)

# iterate over all chromosomes in the infofile
uchroms = np.unique(ifile[:,1])

offset = 0
if '--oneindex' in sys.argv:
    # USE this if the bed file counts from 1-X and not from 0 to X
    offset = 1

flank = 0
if '--add_flanks' in sys.argv:
    flank = int(sys.argv[sys.argv.index('--add_flanks')+1])
    outname += 'flank'+str(flank)

# generate fasta file with sequences
outfasta = open(outname+'.fasta', 'w')
for u, uchr in enumerate(uchroms):
    if os.path.isfile(chrfold.rstrip('/')+'/'+uchr+'.fa.gz'):
        print('Read',chrfold.rstrip('/')+'/'+uchr+'.fa.gz') 
        chrfasta = readfasta(chrfold.rstrip('/')+'/'+uchr+'.fa.gz')
        chrmask = ifile[:,1] == uchr
        print('Generate seq for '+uchr)
        chrmask = np.where(chrmask)[0]
        for e in chrmask:
            extseq = chrfasta[int(ifile[e, 2])-offset-flank:int(ifile[e,3])-offset+flank]
            outfasta.write('>'+ifile[e,0]+'\n'+extseq+'\n')


            
        


    
