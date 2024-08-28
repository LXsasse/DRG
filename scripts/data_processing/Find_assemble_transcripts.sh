# 1) Extract gtf files from all samples individually
PATH/stringtie-2.2.1/stringtie -o ${bamfile%.bam}.gtf -G gencode.vM25.annotation.gtf -m 100 $bamfile
# Relevant arguments
# -f <0.0-1.0>    Sets the minimum isoform abundance of the predicted transcripts as a fraction of the most abundant transcript assembled at a given locus. Lower abundance transcripts are often artifacts of incompletely spliced precursors of processed transcripts. Default: 0.01
# -m <int>        Sets the minimum length allowed for the predicted transcripts. Default: 200
# -a <int>        Junctions that don't have spliced reads that align across them with at least this amount of bases on both sides are filtered out. Default: 10
# -j <float>      There should be at least this many spliced reads that align across a junction (i.e. junction coverage). This number can be fractional, since some reads align in more than one place. A read that aligns in n places will contribute 1/n to the junction coverage. Default: 1
# -g <int>        Minimum locus gap separation value. Reads that are mapped closer than this distance are merged together in the same processing bundle. Default: 50 (bp)

# 2) Merge gtf files with stringtie
ls SK*gtf > all.SK.gtflist.txt
PATH/stringtie-2.2.1/stringtie --merge -o ALL.SK.gtf -G gencode.vM25.annotation.gtf all.SK.gtflist.txt

# Need to insert step to split exon gtf into introns and exons
# Then count both

# 3) Count fragments of each sample to the merged gtf
htseq-count -m union -f bam -t exon -s no $file ALL.SK.gtf > ReCounts/${file%.bam}.ALL.SK.counts.txt

# 4) Determine main transcript in the data sets from coverage and generate .bed files for genes and transcripts
python assigntypeandbed.py ALL.SK.cov.gtf
# Returns: ALL.SK.cov.types.txt, ALL.SK.cov.genes.bed, ALL.SK.cov.exons.bed

# 5) Generate test sets and training lists
python3 makebedsandsets.py ALL.SK.cov.types.txt


