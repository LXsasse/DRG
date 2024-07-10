import numpy as np
import sys,os
from cluster_pwms import read_pwm, read_meme, write_pwm

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



files=sys.argv[1]
if ',' in files:
    files = files.split(',')
else:
    files = [files]

addons=sys.argv[2]
if ',' in addons:
    addons = addons.split(',')
else:
    addons = [addons]

outname = sys.argv[3]

fnames, fpwms = [], []
for f, fi in enumerate(files):
    if os.path.splitext(fi)[-1] == '.meme':
        pwms, names = read_meme(fi)
    else:
        pwms, names = read_pwms(fi)
    
    pwms = np.array([pwm.T for pwm in pwms])
    names = np.array([n+'_'+addons[f] for n in names])
    fnames.append(names)
    fpwms.append(pwms)

fnames = np.concatenate(fnames, axis = 0)
fpwms = np.concatenate(fpwms, axis = 0)

if os.path.splitext(outname)[-1] == '.meme':
    write_meme_file(fpwms, fnames, 'ACGT', outname)
else:
    write_meme_file(outname, fpwms, fnames)




