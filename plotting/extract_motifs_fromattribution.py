import numpy as np
import sys, os
import glob

    
def write_meme_file(pwm, pwmname, alphabet, output_file_path):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = len(pwm)
    print(n_filters)
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= "+alphabet+" \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        meme_file.write("\n")
        meme_file.write("MOTIF %s \n" % pwmname[i])
        meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"% np.count_nonzero(np.sum(pwm[i], axis=0)))

        for j in range(0, np.shape(pwm[i])[-1]):
            for a in range(len(alphabet)):
                if a < len(alphabet)-1:
                    meme_file.write(str(pwm[i][ a, j])+ "\t")
                else:
                    meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()



def find_motifs(a, cut, mg, msig):
    aloc = np.absolute(a)>cut
    sign = np.sign(a) # get sign of effects
    motiflocs = []

    gap = mg +1 # gapsize count
    msi = 1 # sign of motif
    potloc = [] # potential location of motif
    i = 0 
    while i < len(a):
        if aloc[i]: # if location significant
            if len(potloc) == 0: # if potloc just started
                msi = np.copy(sign[i]) # determine which sign the entire motif should have
            
            if sign[i] == msi: # check if base has same sign as rest of motif
                potloc.append(i)
                gap = 0
            elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut: # if the average with gapsize around the location is smaller than the cut then count as gap
                gap += 1
                if gap > mg: # check that gap is still smaller than maximum gap size
                    if len(potloc) >= msig: # if this brought the gapsize over the max gapsize but the motif is long enough, then add to motifs
                        motiflocs.append(potloc)
                    if len(potloc) > 0: # restart where the gap started so that motifs with different direction following directly on other motifs can be counted
                        i -= gap
                    gap = mg + 1
                    potloc = []
        elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut:
            gap +=1
            if gap > mg:
                if len(potloc) >= msig:
                    motiflocs.append(potloc)
                    #print(a[potloc], a[potloc[0]:potloc[-1]])
                if len(potloc) > 0:
                    i -= gap
                gap = mg + 1
                potloc = []
        i += 1
    if len(potloc) >= msig:
        motiflocs.append(potloc)
    return motiflocs


if __name__ == '__main__': 
    
    npzfile = sys.argv[1] # 
    npz = np.load(npzfile, allow_pickle = True)
    names, values, experiments = npz['names'], npz['values'], npz['experiments']
    outname = os.path.splitext(npzfile)[0]
    
    values = np.transpose(values, axes = (0,1,3,2))
    
    seq = np.load(sys.argv[2], allow_pickle = True)
    seqs, snames = seq['seqfeatures'], seq['genenames']
    
    sort = np.argsort(names)[np.isin(np.sort(names), snames)]
    names, values = names[sort], values[sort]
    sort = np.argsort(snames)[np.isin(np.sort(snames), names)]
    snames, seqs = snames[sort], seqs[sort]
    
    cut = float(sys.argv[3])
    maxgap = int(sys.argv[4])
    minsig = int(sys.argv[5])
    norm = sys.argv[6]
    
    outname += '_'+norm+'motifs'+str(cut)+'_'+str(maxgap)+'_'+str(minsig)
    print(outname)
    std = np.sqrt(np.mean(values**2, axis = (-1,-2)))
    if norm == 'condition':
        std = np.mean(std, axis = 0)[None,:, None]
    elif norm == 'seq':
        std = np.mean(std, axis = 1)[:,None, None]
    elif norm == 'global':
        std = np.mean(std)
    else:
        std = np.array(1.)
    
    refatt = np.sum(values*seqs[:,None,:,:], axis = -1)
    stats = refatt/std
    if '--normpwms' in sys.argv and norm not in ['global', 'seq', 'condition']:
        print('normpwms')
        std = np.mean(np.sqrt(np.mean(values**2, axis = (-1,-2))))
        
    
    values = values/std[...,None]
    
    obj = open(outname+'.txt', 'w')
    pwms, pwmnames = [], []
    for n, name in enumerate(names):
        for e, exp in enumerate(experiments):
            motiflocs = find_motifs(stats[n,e], cut, maxgap, minsig)
            for m, ml in enumerate(motiflocs):
                ml = np.array(ml)
                seqname = name+'_'+str(exp)+'_'+str(ml[0])+'-'+str(ml[-1])
                # compute mean, max, loc
                mean = np.mean(refatt[n,e,ml])
                maxs = np.amax(refatt[n,e,ml])
                obj.write(seqname+' '+str(round(mean,3))+' '+str(round(maxs,3))+' '+','.join(ml.astype(str))+'\n')
                pwmnames.append(seqname)
                pwms.append(values[n,e,ml[0]:ml[-1]+1]*np.sign(mean))
    
    np.savez_compressed(outname+'.npz', pwms = pwms, pwmnames = pwmnames)
            
    
    
    
    






    
