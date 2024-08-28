# Combine count outputs from htseq
python combine_counts_tomatrix.py Meandesign.txt --datatypes intronic,exonic --cleancolumns

# Intronic and exonic reads need to be normalized together
# I.e. they are concatenated along axis 0 to create a matrix of length (2Xgenes)XCelltypes
# TPM is generally used to be able to do differential expression analysis, i.e understand differences between cells
# https://hbctraining.github.io/DGE_workshop/lessons/02_DGE_count_normalization.html
# USE gtf with exons and introns only (no filtering for canonical transcripts)
python simple_data_processing.py Meandesign.intronic.mat.tsv Meandesign.exonic.mat.tsv --TPM Mus_musculus.GRCm38.100.chr.gtf_2023-04-07_constExonsAndIntrons_UCSC.gtf --transform mean,add=1,log2

# Align gene names and craete log2 change between intron and exon as Kd
python makekd.py Meandesign.intronic.mat.tpm.mean.add1.log2.tsv Meandesign.exonic.mat.tpm.mean.add1.log2.tsv Meandesign.degradation.mat.tpm.mean.add1.log2.tsv


