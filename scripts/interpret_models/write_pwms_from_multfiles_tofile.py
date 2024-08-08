import numpy as np
import sys,os
from drg_tools.io_utils import readin_motif_files, write_pwm, write_meme_file

if __name__ == '__main__':

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
        pwms, names, nts = readin_motif_files(fi)
        print(len(pwms), np.shape(pwms[0]))
        print(np.shape(pwms))
        pwms = [pwm.T for pwm in pwms]
        names = np.array([n+'_'+addons[f] for n in names])
        fnames.append(names)
        fpwms.append(pwms)

    fnames = np.concatenate(fnames, axis = 0)
    fpwms = np.concatenate(fpwms, axis = 0)

    if os.path.splitext(outname)[-1] == '.meme':
        write_meme_file(fpwms, fnames, 'ACGT', outname)
    else:
        write_meme_file(outname, fpwms, fnames)




