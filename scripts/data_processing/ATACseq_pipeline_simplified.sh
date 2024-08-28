fastq_r1=1002-Effector_CD4pos_T-S_R1_001.fastq.gz
fastq_r2=1002-Effector_CD4pos_T-S_R2_001.fastq.gz

base_name=${fastq_r1%_R1_001.fastq.gz}

fasta_file=/home/sheddn/genome/hg38/hg38.fa

# Trim adapters
java -jar /software/trimmomatic-0.38.jar PE -threads {threads} -phred33 $fastq_r1 $fastq_r2 ${base_name}.R1.P.fastq.gz ${base_name}.R1.U.fastq.gz ${base_name}.R2.P.fastq.gz  ${base_name}.R2.U.fastq.gz LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36

# Map with Bowtie2
bowtie2 -p {threads} -x $fasta_file -1 ${base_name}.R1.P.fastq.gz -2 ${base_name}.R2.P.fastq.gz > ${base_name}.sam

# sort
samtools view -b -@ {threads} ${base_name}.sam | samtools sort -@ {threads} -o ${base_name}.bam -

#Calculating mapping statistics
samtools flagstat -@ {threads} ${base_name}.bam > ${base_name}.flagstat

# Filter alignment and convert to BAM
samtools view -@ {threads} -q 30 -F4 -O SAM ${base_name}.bam | egrep -v chrM | samtools view -b -@ {threads} -o ${base_name}.filtered.bam -T $fasta_file -

### Won't need to call peaks
# Call peaks
#macs2 callpeak -t ${base_name}.bam  --nolambda --nomodel -g hs --keep-dup all --call-summits -n ${base_name}

# Generate bigwig
#bamCoverage -b ${base_name}.bam -o ${base_name}.bigWig --extendReads --ignoreForNormalization chrX -bs 10 -p {threads} --normalizeUsing RPKM

