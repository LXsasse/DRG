# RNAseq alignment for paired-end reads
 STAR --genomeDir $GENOME_INDEX --sjdbGTFfile $GTF_FILE --readFilesIn ${sample_ID}.R1.fastq.gz ${sample_ID}.R2.fastq.gz --readFilesCommand zcat --runThreadN $CORE --outSAMtype BAM SortedByCoordinate --alignSJDBoverhangMin 1 --outFilterMismatchNoverLmax 0.05 --outFilterScoreMinOverLread 0.90 --outFilterMatchNminOverLread 0.90 --alignIntronMax 1000000
mv Aligned.sortedByCoord.out.bam ${sample_ID}.sortedByCoord.bam
samtools index ${sample_ID}.sortedByCoord.bam
#The genome assembly and annotation are from GENCODE (https://www.gencodegenes.org/mouse/releases.html). Both of them are primary regions (PRI).
#In our current pipeline, we still use M25 based on GRCm38.p6. We've planned to move forward with GRCm39, but it will be a big job for us to re-map all our previous batches of data. 
#Genome sequence, primary assembly (GRCm38): https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz
#Comprehensive gene annotation: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.primary_assembly.annotation.gtf.gz
#The sjdbOverhang option needs to be specified for detecting possible splicing sites. It usually equals the read length minus 1. In our ULI-RNAseq data, since the read length is 38 bases, we set up this parameter to 37. 
# IMPORTANT: READ methods to find out the correct value for sjdOverhang. Yoshida 2019 was 25-1

