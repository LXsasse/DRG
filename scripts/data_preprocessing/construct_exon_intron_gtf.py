import numpy as np
import sys, os
import gzip 
from functools import reduce 

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
    evidence = []
    strand = []
    gene_id, gene_type, gene_name, trans_id = [], [], [], []
    
    for l, line in enumerate(fi):
        if line[0] != '#':
            line = line.strip().split('\t')
            chrom.append(line[0]) # chromosomes
            evidence.append(line[1]) # resource from which genes were taken, like Havana or Ensembl
            itype.append(line[2]) # type of sequence: i.e. exon, transcript, gene
            start.append(int(line[3])) # 
            end.append(int(line[4]))
            strand.append(line[6]) 
            info = line[8].split(';')
            gid, gty, gna, gtr = '' ,'', '', '' # gene_id, gene_type, gene_name, transcript_id
            for i, inf in enumerate(info):
                inf = inf.strip()
                if inf[:7] == 'gene_id':
                    inf = inf.split()
                    gid = inf[1].strip('"')
                if inf[:13] == 'transcript_id':
                    inf = inf.split()
                    gtr = inf[1].strip('"')
                if inf[:9] == 'gene_type':
                    inf = inf.split()
                    gty = inf[1].strip('"')
                if inf[:9] == 'gene_name':
                    inf = inf.split()
                    gna = inf[1].strip('"')
            gene_id.append(gid)
            gene_name.append(gna)
            gene_type.append(gty)
            trans_id.append(gtr)
    return np.array([chrom, start, end, strand, evidence, itype, gene_id, gene_type, gene_name, trans_id], dtype = object).T
    
def determine_introns(transcript, exons):
    introns = []
    if transcript[0] != exons[0][0]:
        introns.append([transcript[0], exons[0][0]-1])
    for e in range(len(exons)-1):
        introns.append([exons[e][-1]+1,exons[e+1][0]-1])
    if transcript[-1] != exons[-1][-1]:
        introns.append([exons[-1][-1]+1, transcript[-1]])
    return introns

def determine_constitutive_introns(intronsets, transcriptset):
    const_transcript = []
    for trse in transcriptset:
        #print(trse)
        const_transcript.append(np.arange(trse[0], trse[1], dtype = int))
    const_transcript = reduce(np.union1d, const_transcript)
    #print(const_transcript)
    const_introns = []
    for i, intset in enumerate(intronsets):
        #print(intset)
        ints = [np.arange(its[0], its[1]+1, dtype = int) for its in intset]
        if len(ints) > 0:
            ints = np.concatenate(ints)
        ints = np.append(ints, np.setdiff1d(const_transcript, np.arange(transcriptset[i][0], transcriptset[i][1], dtype = int)))
        const_introns.append(ints)
    const_introns = reduce(np.intersect1d,const_introns)
    #print(const_introns)
    diff = np.where(np.diff(const_introns) > 1)[0]
    constitutive_introns = []
    #print(diff, constitutive_introns)
    for d, di in enumerate(diff):
        if d == 0:
            constitutive_introns.append([const_introns[0], const_introns[di]])
        else:
            constitutive_introns.append([const_introns[diff[d-1]+1], const_introns[di]])
        if d == len(diff)-1:
            constitutive_introns.append([const_introns[diff[d]+1], const_introns[-1]])
    #print(constitutive_introns)
    # iterate over transcripts and only cosider constitutive introns within that transcript
    indiv_const_introns = []
    for trse in transcriptset:
        trans_introns =[]
        for const_int in constitutive_introns:
            if const_int[0] >= trse[0] and const_int[1] <= trse[1]:
                trans_introns.append(const_int)
        indiv_const_introns.append(trans_introns)
                
    return indiv_const_introns
    
    
if __name__ == '__main__':    

    infofile = sys.argv[1] # gtffile that contains all the locations for the regions

    outname = os.path.splitext(sys.argv[1].strip('.gz'))[0] + '.Exons'
    if '--constitutive_introns' in sys.argv:
        outname += '.constIntrons'
    else:
        outname += '.Introns'


    ifile = readgtf(infofile)
    print(np.shape(ifile))

    if '--source' in sys.argv:
        source = sys.argv[sys.argv.index('--source')+1]
        mask =ifile[:,4] == source
        ifile = ifile[mask]
        outname += '.'+source
        print('source', np.shape(ifile))
        
    if '--chrtype' in sys.argv:
        seqtype = sys.argv[sys.argv.index('--chrtype')+1]
        if ',' in seqtype:
            seqset = seqtype.split(',')
        elif '-' in seqtype:
            seqset = seqtype.split('.')
            seqset = [seqset[0]+str(i) for i in range(int(seqset[1].split('-')[0]), int(seqset[1].split('-')[1])+1)]
        else:
            seqset = [seqtype]
        mask = np.isin(ifile[:,0], seqset)
        ifile = ifile[mask]
        outname += '.'+seqtype
        print(seqset, np.shape(ifile))

    outname += os.path.splitext(sys.argv[1].strip('.gz'))[1]
    print(outname)

    transcripts = np.where(ifile[:,5] == 'transcript')[0]
    exons = np.where(ifile[:,5] == 'exon')[0]

    unique_transcripts = np.unique(ifile[transcripts,6])
    obj = open(outname, 'w')
    
    for v, ut in enumerate(unique_transcripts):
        #print(ut)
        alltrans =  transcripts[np.where(ifile[transcripts,6] == ut)[0]]
        #print(alltrans)
        allex = exons[np.where(ifile[exons,6] == ut)[0]]
        #print(allex)
        allintrons = []
        alltranscripts = []
        for t, trans in enumerate(ifile[alltrans, -1]):
            tr = alltrans[t]
            transloc = ifile[tr][[1,2]]
            alltranscripts.append(transloc)
            #print(transloc)
            # determine introns of transcript
            specex = allex[ifile[allex,-1] == trans]
            exonloc = ifile[specex][:,[1,2]]
            introns = determine_introns(transloc, exonloc)
            #print(exonloc, introns)
            allintrons.append(introns)
        if '--constitutive_introns' in sys.argv and len(allintrons) > 1:
            # determine constitutive introns of transcript
            constitutive_introns = determine_constitutive_introns(allintrons, alltranscripts)
        else:
            constitutive_introns = allintrons
        
        for t, trans in enumerate(ifile[alltrans, -1]):
            specex = allex[ifile[allex,-1] == trans]
            tr = alltrans[t]
            for s, sp in enumerate(specex):
                obj.write(str(ifile[sp,0])+'\t'+str(ifile[sp,4])+'\t'+str(ifile[sp,5])+'\t'+str(ifile[sp,1])+'\t'+str(ifile[sp,2])+'\t.\t'+str(ifile[sp,3])+'\t.\t'+' gene_id ' +str(ifile[sp,-4])+'; gene_type '+str(ifile[sp,-3])+'; gene_name '+str(ifile[sp,-2])+'; transcript_id '+str(ifile[sp,-1])+'\n')
            for c, con_ex  in enumerate(constitutive_introns[t]):
                if len(con_ex) > 0:
                    obj.write(str(ifile[tr,0])+'\t'+str(ifile[tr,4])+'\t'+'intron'+'\t'+str(con_ex[0])+'\t'+str(con_ex[1])+'\t.\t'+str(ifile[tr,3])+'\t.\t'+' gene_id ' +str(ifile[tr,-4])+'; gene_type '+str(ifile[tr,-3])+'; gene_name '+str(ifile[tr,-2])+'; transcript_id '+str(ifile[tr,-1])+'\n')
        
                
            
        #[chrom, start, end, strand, evidence, itype, gene_id, gene_type, gene_name, trans_id]
        
    
        
        
        
    
    
    




    
