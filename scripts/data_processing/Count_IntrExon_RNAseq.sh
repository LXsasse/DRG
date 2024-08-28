# Create intronic and exonic count files

## Generate gtf with introns with CRIES (https://github.com/csglab/CRIES)
### This worked only if we download gtf file from Ensembl https://ftp.ensembl.org/pub/release-100/gtf/mus_musculus/ (might works with encode gtf if you change chromosome name
Rscript PATH/CRIES-master/generate_const_intronsExons_gtf.R Mus_musculus.GRCm38.100.chr.gtf UCSC .

### Otherwise, use own script
python create_exon_intron_gtf.py gencode.vM25.primary_assembly.annotation.gtf --source HAVANA --chrtype chr.1-19 --constitutive_introns

## Use HTSeq to count intronic and exonic reads (Install HTSeq :pip install HTSeq)
htseq-count -m intersection-strict -f bam -t exon -s no SK-3L4G_A4.sortedByCoord.bam ./Mus_musculus.GRCm38.100.chr.consExons.gtf > SK-3L4G_A4.sortedByCoord.consExons.counts.txt
htseq-count -m union -f bam -t intron -s no SK-3L4G_A4.sortedByCoord.bam /Mus_musculus.GRCm38.100.chr.Introns.gtf > SK-3L4G_A4.sortedByCoord.Introns.counts.txt
### The parameter <strand> depends on the strandedness of the library preparation kit. For Illumina TruSeq RNA Library Prep Kit, the correct parameter is often reverse. If not sure, try all the three different options no, yes and reverse, and decide accordingly. Note that for paired-end reads, -r name must be used along with BAM files that are sorted by read name.

### You can use variance-stabilized transformation of DESeq or DESeq2 for read count normalization (for each set of intronic and exonic reads separately), and then take Δexon–Δintron between any two samples as the measure of differential stability. However, note that this measure can be biased, overestimating the stability of transcriptionally down-regulated genes and underestimating the stability of transcriptionally up-regulated genes, as discussed in this paper: Alkallas et al., Nat Commun, 8:909.

