# Process data for model training

Data processing scripts are located all located in:

```
prscripts=/path/to/scripts/data_preprocessing/
```

## Download genome files from UCSC genome browser

```
genomedir=mm10
mkdir $genomedir
cd $genomedir

for i in {1..19}
do
wget --timestamping 'ftp://hgdownload.cse.ucsc.edu/goldenPath/mm10/chromosomes/chr'${i}'.fa.gz' -O chr${i}.fa.gz
done
cd ..
```

## Extract sequences from bed, gtf, or other file. 

From start to end defined in bed file
```
bedfile=Test.bed
python ${prscripts}generate_fasta_from_bedgtf_and_genome.py $genomedir $bedfile 
```
If sequences have different lenght and they should all be set to 1000bp
```
python ${prscripts}generate_fasta_from_bedgtf_and_genome.py $genomedir $bedfile --extend_to_length 1000
```

Extract sequences around the TSS. The TSS is selected based of the strand, i.e. start if + and end if -
```
python ${prscripts}generate_fasta_from_bedgtf_and_genome.py $genomedir $bedfile --from_tss --add_flanks 500
```

If you're working with a gtf file and want to extract transcripts
```
gtfile=Test.gtf
python ${prscripts}generate_fasta_from_bedgtf_and_genome.py $genomedir $gtffile --seqtype --genetype 'constitutive' --generate_transcripts
```


## From the generated fasta file, generate a one hot encoding

```
fasta=Test.fasta
python ${prscripts}transform_seqtofeature.py $fasta --cut_seqlength 40000
# if anything is longer than 40kb, reduce it to 40kb
```

This program can also generate k-mer representations

```
python ${prscripts}transform_seqtofeature.py $fasta --kmers regular # gapped, mismatch, decreasing
```

The script can also add genomic annotations in another channel, e.g. CDS, UTR, gene_body, etc. 
```
regfile=genomic_regions.txt # genomic regions are provided as Gene_name, region_name, location, total_length_of_gene
python ${prscripts}transform_seqtofeature.py $fasta --genomicregions 
```

Sometimes it might be wise to align sequence of different length to their end instead of their start (default left), or their center, to improve the models abilty to learn from positions in sequence

```
python ${prscripts}transform_seqtofeature.py $fasta --align_sequence right
```

## Create train, test and validation set.

Lastly, to avoid data leakage and compare models between trainings or architectures, we should pre-define train, test, and validation sets based on chromosomes. 
this can either be the bed or the gtf file 
```
python ${prscripts}generate_chromosome_testtrainsetfolds_frombegorgtf.py $bedfile --exclude chrY,chrX 
```

