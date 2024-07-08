import numpy as np
import sys, os

# check if string can be integer or float
def numbertype(inbool):
    try:
        int(inbool)
    except:
        pass
    else:
        return int(inbool)
    try:
        float(inbool)
    except:
        pass
    else:
        return float(inbool)
    return inbool



# Read text files with PWMs
def read_pwm(pwmlist, nameline = 'Motif'):
    names = []
    pwms = []
    pwm = []
    obj = open(pwmlist, 'r').readlines()
    for l, line in enumerate(obj):
        line = line.strip().split('\t')
        if ((len(line) == 0) or (line[0] == '')) and len(pwm) > 0:
            pwm = np.array(pwm, dtype = float).T
            pwms.append(np.array(pwm))
            pwm = []
            names.append(name)
        elif len(line) > 0:
            if line[0] == nameline:
                name = line[1]
                pwm = []
            elif line[0] == 'Pos':
                nts = line[1:]
            elif isinstance(numbertype(line[0]), int):
                pwm.append(line[1:])
    return pwms, names, nts
    
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
        if np.sum(pwm[i]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF %s \n" % pwmname[i])
            meme_file.write(
                "letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"
                % np.count_nonzero(np.sum(pwm[i], axis=0))
            )
        
        for j in range(0, np.shape(pwm[i])[-1]):
            if np.sum(pwm[i][:, j]) > 0:
                for a in range(len(alphabet)):
                    if a < len(alphabet)-1:
                        meme_file.write(str(pwm[i][ a, j])+ "\t")
                    else:
                        meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()
    
    
if __name__ == '__main__':
    
    pwmfile = sys.argv[1]
    pwms, names, nts = read_pwm(pwmfile)
    outname = os.path.splitext(pwmfile)[0]
    
    if '--filtersize' in sys.argv:
        sizefile = np.genfromtxt(sys.argv[sys.argv.index('--filtersize')+1], dtype = str)
        minsize = int(sys.argv[sys.argv.index('--filtersize')+2])
        cl, ncl = np.unique(sizefile[:,1].astype(int), return_counts = True)
        mask = np.where(ncl >= minsize)[0]
        outname += 'ms'+str(minsize)
        pwms, names = [pwms[i] for i in mask], [names[i] for i in mask]
    
    if '--set' in sys.argv:
        setfile = sys.argv[sys.argv.index('--set')+1]
        tset = np.genfromtxt(setfile, dtype = str)
        mask = np.where(np.isin(names, tset))[0]
        outname += '_'+os.path.splitext(os.path.split(setfile)[1])[0]
        pwms, names = [pwms[i] for i in mask], [names[i] for i in mask]
        outname += 'sbst'+str(len(names))
        
    if '--adjust_sign' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.sign(np.sum(pwm[np.argmax(np.absolute(pwm),axis = 0),np.arange(len(pwm[0]),dtype = int)]))*pwm
    
    if '--exppwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.exp(pwm)
        
    if '--norm' in sys.argv or '--normpwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.sum(pwm,axis =0)
    
    if '--infocont' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    
    if '--changenames' in sys.argv:
        clusters = np.arange(len(names)).astype(str)
    else:
        clusters = names
    outname += '.meme'
    write_meme_file(pwms, clusters, ''.join(nts), outname)
    
    
    
    
    
