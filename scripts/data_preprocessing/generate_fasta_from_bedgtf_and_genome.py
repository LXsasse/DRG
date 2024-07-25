import numpy as np
import sys, os
import gzip 


from drg_tools.io_utils import readgtf, readgenomefasta, reverse_complement_seqstring



if __name__ == '__main__':
        
    chrfold = sys.argv[1] # folder with all chromosomes in it
    infofile = sys.argv[2] # gtffile that contains all the locations for the regions

    outname = sys.argv[1].strip('/').split('/')[-1]+os.path.splitext(os.path.split(sys.argv[2].strip('.gz'))[1])[0]
    print(outname)

    # bedfile order:
    #chr start end name number_exon strand
    if os.path.splitext(infofile)[-1] == '.bed':
        ifile = np.genfromtxt(infofile, dtype = str)
    elif os.path.splitext(infofile)[-1] == '.csv' or os.path.splitext(infofile)[-1] == '.txt':
        ifile = np.genfromtxt(infofile, dtype = str)
        ifile = ifile[:,[1,2,3,0]]
    else:
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
        names = ifile[:,-1]
    elif '--usegeneid_noversion':
        vnames = ifile[:,-1]
        names = []
        for v, vn in enumerate(vnames):
            if '.' in vn:
                vn = vn.split('.')[0]
            names.append(vn)
        names = np.array(names)
    else:
        names = ifile[:,3]


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
            chrfasta = readgenomefasta(chrfold.rstrip('/')+'/'+uchr+'.fa.gz')
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
                            ext = reverse_complement_seqstring(ext)
                            extseq = ext + extseq
                        else:
                            extseq = extseq + ext
                    outfasta.write('>'+na+'\n'+extseq+'\n')
                elif '--from_tss' in sys.argv:
                    e = chrmask[0]
                    
                    if ifile[e, 5] == '-':
                        extseq = chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                        extseq = reverse_complement_seqstring(extseq)
                        outfasta.write('>'+na+'\n'+extseq+'\n')
                    elif ifile[e, 5] == '+':
                        extseq = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                        outfasta.write('>'+na+'\n'+extseq+'\n')
                    elif ifile[e, 5] == '.':
                        extseqf = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,1])-offset+flank+flank]
                        extseqb= chrfasta[int(ifile[e, 2])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                        extseqb = reverse_complement_seqstring(extseq)
                        outfasta.write('>'+na+'\n'+extseqf+'\n'+'>'+na+'.rev\n'+extseqb+'\n')
                else:
                    e = chrmask[0]
                    extseq = chrfasta[int(ifile[e, 1])-offset-flank+flank:int(ifile[e,2])-offset+flank+flank]
                    if ifile[e, 5] == '-':
                        extseq = reverse_complement_seqstring(extseq)
                    outfasta.write('>'+na+'\n'+extseq+'\n')
            
            
        


    
