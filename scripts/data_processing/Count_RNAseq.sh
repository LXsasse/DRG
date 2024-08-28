# Create bam files
./RNAseq_pipeline_simplified.sh


# Craete count files for RNA-seq (IGNORE if you want to quantify intronic and exonic counts)

## Download gtf from https://www.gencodegenes.org/mouse/release_M25.html
### We are using the comprehensive PRI annotation https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.primary_assembly.annotation.gtf.gz

### We use featureCounts for RNAseq reads quantification (https://subread.sourceforge.net/SubreadUsersGuide.pdf; P32). The parameters are shown below. Here, '-t exon' means only rows which have a matched feature type (exon) in the provided GTF annotation file will be included for read counting. It counts the reads mapped to all the exons from the same gene, no matter which transcript they are from.

featureCounts -p -B -C -T 4 -F GTF -a gencode.vM25.primary_assembly.annotation.gtf -g gene_name -s 0 -t exon --minOverlap 5 --ignoreDup -o featureCounts.txt Sample.sortedByCoord.bam

