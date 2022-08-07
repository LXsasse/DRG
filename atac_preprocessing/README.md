Use these scripts to preprocess ATAC seq data and get count files of various resolutions.
Examples and description:
  atac_counts.py:
    Uses bam and bed file to compute the coverage within the regions in the bed file. Can shift the reads or fragments and count individual ends or over the entire fragment. Returns npz file with 'names' of the regions from bed file, 'counts' within that region. Counts are of shape (names,1,bins_in_window) to concatenate along axis one later.
    E.g. python atac_counts.py sorted.GN.Thio.PC.bam ImmGenATAC1219.peak_matched.txt --shift 4,-4 --outdir BPprofiles/ --countmode 53
  bigwigtonpz.py:
    Uses bigwig file and bed file to generate npz file for coverage within the defined regions in bed file.
    E.g. python bigwigtonpz.py final.GN.Thio.PC.unstranded.bw ImmGenATAC1219.peak_matched.txt
