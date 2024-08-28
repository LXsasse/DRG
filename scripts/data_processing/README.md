# Currently for Transcriptional process aware (TPA) CNN model, we use the following steps 

## Create intronic and exonic gtf files and count coverage
```
./Count_IntrExon_RNAseq.sh
```

## Process count files as described in 
```
Process_intron_exon_counts.sh
```

If you want to determine differentially expressed, or differentially processed genes, follow 
```
Differential_analysis_RNAseq.sh
```
