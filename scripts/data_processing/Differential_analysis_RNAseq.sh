# DiffRAC performs differential processing rate analysis with DESeq2
## Unfortunately, DESeq2 does not converge for large count matrices with many different cell types

#Rscript run_DiffRAC.R Meandesign.txt Counts/ Blind Mean NormCounts/ALLtoMEAN
# Took the sum of all cells according to the distribution of cells and conditions
#Rscript run_DiffRAC.R SUMdesign.txt Counts/ Blind Mean NormCounts/ALLtoSUM

## Usually, we just this to determine genes with significant changes
### Design matrix files
indesign='Bfodesign.txt MCdesign.txt MFrpdesign.txt NKdesign.txt T8emdesign.txt DC8design.txt Modesign.txt pDCdesign.txt Tgddesign.txt MFpcdesign.txt MZBdesign.txt T4ndesign.txt Tregdesign.txt'

for i in $indesign
do
Rscript run_DiffRAC.R $i Counts/ Blind ToCTRL NormCounts/${i%design.txt}toIL
Rscript DESeqTsv.R $i $countmatrix $outname ToCTRL 
done





