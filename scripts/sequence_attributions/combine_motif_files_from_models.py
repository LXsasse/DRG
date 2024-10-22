import numpy as np
from drg_tools.io_utils import readin_motif_files, write_meme_file
import sys, os
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='combine_meme',
                    description='Combines motifs from different motif files with different prefixes for their names')
    parser.add_argument('pwmfiles', type=str, 
                        help='Names of pwms files, separated by ,. This can be a meme, a txt, or npz file with pwms and pwmnames, OR the .npz file with the stats from previous clustering')
    parser.add_argument('--prefixes', type=str, 
                        help='Prefixes for motifs from different files separated by ","', default = None)
    parser.add_argument('--npz', action='store_true', 
                        help='returns npz instead of meme')
    parser.add_argument('--outname', type = str, default = None)
    
    args = parser.parse_args()
    
    pwmfiles = args.pwmfiles.split(',')
    
    if args.prefixes is not None:
        prefixes = args.prefixes.split(',')
    else:
        prefixes = np.arange(len(pwmfiles), dtype = int).astype(str)
    
    pwms, pnames = [], []
    for p, pwmfile in enumerate(pwmfiles):
        pwms1, pnames1, nts1 = readin_motif_files(pwmfile)
        pnames1 = np.array([prefixes[p]+'_'+name for name in pnames1])
        pwms.append(pwms1)
        pnames.append(pnames1)
    
    pwms = np.concatenate(pwms, axis = 0)
    pwmnames = np.concatenate(pnames, axis = 0)
    
    if args.outname is None:
        args.outname = '-'.join(np.array(prefixes))+'_'+os.path.splitext(os.path.split(pwmfiles[0])[1])[0]
    
    if args.npz:
        np.savez_compressed(args.outname+'.npz', pwms = pwms, pwmnames = pwmnames)
    else:
        write_meme_file(pwms, pwmnames, 'ACGT', args.outname+'.meme')

