#---
#title: Application of DiffRAC
#author:
#- name: Hamed S. Najafabadi
#- name: Gabrielle Perron

library(DESeq2)
library(lineup2)
source("~/UW/Software/DiffRAC-master/DiffRAC.R")

args <- commandArgs(trailingOnly = TRUE)

tablePath <- args[1]	# design matrix and location of individual htcount files
inputFolder <- args[2]	# location of folder from which tablePath goes individual files
normalization <- args[3] # if Blind perform blind dispersion calculation, if Batch normalize batch, if Condition normalize with Condition, if BatchCondition normalize batch and condition
method <- args[4]	# Mean or anything else (Control), Mean computes differential expression to mean off all conditions
outname <- args[5]	# Path and name of output

sampleTable <- read.csv( tablePath, sep="\t" ) # Read design and file paths

# Iterate over two types of read counts
for( readType in c("exonic","intronic") )
{
        print( paste("Analyzing ", readType, " reads ...", sep="") )
	# Read in DEseq object from all files in the sampleTable
        cds <- DESeqDataSetFromHTSeqCount( # omits special rows from htseq-count
					 sampleTable = sampleTable[sampleTable$ReadType==readType,],
					 directory = inputFolder,
					 design = ~ 1 # required parameter, 1 means that there's only a bias term in the design matrix
	)
	if( readType == 'exonic')
	{	exon_counts <- as.matrix(counts(cds))
		coldata_exon <- colData(cds)
	} else
	{	intron_counts <- as.matrix(counts(cds))
		coldata_intron <-  colData(cds)
	}

}


exon_counts <- exon_counts[rowSums(exon_counts[]) > 0,] # Remove all rows with exclusively 0
intron_counts <- intron_counts[rowSums(intron_counts[]) > 0,] 
comb_counts <- align_matrix_rows(exon_counts, intron_counts) # Align gene rows between exon and introns
exon_counts <- comb_counts$x
intron_counts <- comb_counts$y
ide <- which(exon_counts == NA)	# Replace NA with 0
idi <- which(intron_counts == NA)
exon_counts[ide] <- 0
intron_counts[idi] <- 0
print(nrow(exon_counts))
print(nrow(intron_counts))
#print(head(exon_counts, 10))
#print(head(intron_counts,10))

# create new DESeq object with aligned and cleaned counts
if ( normalization == 'Batch'){
	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ Batch)
	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ Batch)
} else if (normalization == 'Condition'){
	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ Condition)
	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ Condition)
} else if (normalization == 'BatchCondition') {
	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ Batch + Condition)
	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ Batch + Condition)
} else {
	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ 1)
	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ 1)
}

# Execute DESeq here ??? Wondering if this changes outcome of vst???
#print('Design')
#print(design(cds_exon))
#dds_exon <- DESeq(cds_exon)
#dds_intron <- DESeq(cds_intron)


# Normalize the mean and the variance of the counts
# Should be similar to estimateSizeFactor and then estimateDispersion
# estimateSizeFactor corrects for number of counts in sample
# estimateDispersion corrects for variance increase for low counts
# can also use VarianceStabilizingTranformation
# vst is the same VarianceStabilizingTransformation but uses subset of samples to estimate dispersion to decrease run time
if(normalization == 'Blind'){
	vsd_exon <- vst(object = dds_exon, blind = T)
	vsd_intron <- vst(object = dds_intron, blind = T)
} else {
	vsd_exon <- vst(object = dds_exon, blind = F)
        vsd_intron <- vst(object = dds_intron, blind = F)
}

print('VST')
print(head(assay(vsd_exon),5))

# Remove batch effect from count data
if (normalization == 'Batch' || normalization == 'BatchCondition'){
	print( 'Removing batch effects from exonic and intronic counts')
	mat <- assay(vsd_exon)
	mat <- limma::removeBatchEffect(mat, vsd_exon$Batch)
	assay(vsd_exon) <- mat
	mat <- assay(vsd_intron)
        mat <- limma::removeBatchEffect(mat, vsd_intron$Batch)
        assay(vsd_intron) <- mat
	print('Debatched')
	print(head(assay(vsd_exon), 5))
}


write.table(assay(vsd_exon), paste(outname,"_normalized.",normalization,".exonic.mx.txt",sep=""),quote=F, sep="\t")
write.table(assay(vsd_intron), paste(outname,"_normalized.",normalization,".intronic.mx.txt",sep=""),quote=F, sep="\t")




# Perform enrichment analysis for exons and introns indepdently
if( TRUE ){
	if ( normalization == 'Batch' || normalization == 'BatchCondition'){
        	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ Batch + Condition)
        	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ Batch+ Condition)
	} else {
        	cds_exon <- DESeqDataSetFromMatrix(countData = exon_counts, colData = coldata_exon, design = ~ Condition)
        	cds_intron <- DESeqDataSetFromMatrix(countData = intron_counts, colData = coldata_intron, design = ~ Condition)
	}
	
	print(design(cds_exon))
	print(design(cds_intron))

	# Execute DESeq here ??? Wondering if this changes outcome of vst???
	dds_exon <- DESeq(cds_exon)
	dds_intron <- DESeq(cds_intron)

	exon_ratnames <- resultsNames(dds_exon)
	intron_ratnames <- resultsNames(dds_intron)
	print(exon_ratnames)
	# Number of columns in results
	nce <- length(exon_ratnames)
	nci <- length(intron_ratnames)
	# Find indexes of columns with 'Condition' in them
	exon_cons <- grep("Condition", exon_ratnames)
	intron_cons <- grep("Condition", intron_ratnames)
	print(exon_cons)
	nceall <- length(exon_cons)
	nciall <- length(intron_cons)

	i <- 1
	j <- 1
	if(method == 'Mean') { 
		# Mean method return differntial expression to the mean of all conditions
        	log2df_exon <- data.frame(matrix(NA, nrow = nrow(dds_exon), ncol = nceall+1)) # initialize empty data frames
        	pdf_exon <- data.frame(matrix(NA, nrow = nrow(dds_exon), ncol = nceall+1))
		log2df_intron <- data.frame(matrix(NA, nrow = nrow(dds_intron), ncol = nciall+1)) # initialize empty data frames
                pdf_intron <- data.frame(matrix(NA, nrow = nrow(dds_intron), ncol = nciall+1))

        	contrast_exon <- numeric(nce) # contrast method can return differential expression for combination of colums that are in the design matrix
		contrast_intron <- numeric(nci)
        	contrast_exon[exon_cons] <- -1/(nceall+1)
		contrast_intron[intron_cons] <- -1/(nciall+1)
        
		res_exon <- as.data.frame( results(dds_exon, contrast = contrast_exon) )
		res_intron <- as.data.frame( results(dds_intron, contrast = contrast_intron) )
        	colnames(log2df_exon)[i] <- 'Control_vs_Mean'
        	colnames(pdf_exon)[i] <- 'Control_vs_Mean'
		colnames(log2df_intron)[i] <- 'Control_vs_Mean'
                colnames(pdf_intron)[i] <- 'Control_vs_Mean'

        	log2df_exon[,i] <- res_exon[,'log2FoldChange']
       		pdf_exon[,i] <- res_exon[,'pvalue']
		log2df_intron[,i] <- res_intron[,'log2FoldChange']
                pdf_intron[,i] <- res_intron[,'pvalue']
        	i <- i+1
		j <- j+1
        	for(a in exon_cons)
        	{
                	contrast_exon <- numeric(nce)
                	contrast_exon[exon_cons] <- -1/(nceall+1)
                	contrast_exon[a] <- contrast_exon[a] + 1
                	print(exon_ratnames[a])
                	cname <- paste(sapply(strsplit(exon_ratnames[a], '_'), '[', 2),'_vs_Mean',sep='')
			print(cname)
                	res <- as.data.frame( results(dds_exon, contrast = contrast_exon) )
                	rownames(log2df_exon) <- rownames(res)
                	rownames(pdf_exon) <- rownames(res)
                	colnames(log2df_exon)[i] <- cname
                	colnames(pdf_exon)[i] <- cname
                	log2df_exon[,i] <- res[,'log2FoldChange']
                	pdf_exon[,i] <- res[,'pvalue']
                	i <- i+1

        	}
		for(a in intron_cons)
                {
                        contrast_intron <- numeric(nci)
                        contrast_intron[intron_cons] <- -1/(nciall+1)
                        contrast_intron[a] <- contrast_intron[a] + 1
                        print(intron_ratnames[a])
                        cname <- paste(sapply(strsplit(intron_ratnames[a], '_'), '[', 2), '_vs_Mean', sep='')
                        res <- as.data.frame( results(dds_intron, contrast = contrast_intron) )
                        rownames(log2df_intron) <- rownames(res)
                        rownames(pdf_intron) <- rownames(res)
                        colnames(log2df_intron)[j] <- cname
                        colnames(pdf_intron)[j] <- cname
                        log2df_intron[,j] <- res[,'log2FoldChange']
                        pdf_intron[,j] <- res[,'pvalue']
                        j <- j+1
		}

	} else {
		log2df_exon <- data.frame(matrix(NA, nrow = nrow(dds_exon), ncol = nceall+1)) # initialize empty data frames
                pdf_exon <- data.frame(matrix(NA, nrow = nrow(dds_exon), ncol = nceall+1))
                log2df_intron <- data.frame(matrix(NA, nrow = nrow(dds_intron), ncol = nciall+1)) # initialize empty data frames
                pdf_intron <- data.frame(matrix(NA, nrow = nrow(dds_intron), ncol = nciall+1))
		i <- 1
                j <- 1
                for(a in exon_cons)
                {
                        contrast_exon <- numeric(nce)
                        contrast_exon[a] <- 1
                        print(exon_ratnames[a])
                        cname <- sub('Condition_', '', exon_ratnames[a])
			print(cname)
                        res <- as.data.frame( results(dds_exon, contrast = contrast_exon) )
                        rownames(log2df_exon) <- rownames(res)
                        rownames(pdf_exon) <- rownames(res)
                        colnames(log2df_exon)[i] <- cname
                        colnames(pdf_exon)[i] <- cname
                        log2df_exon[,i] <- res[,'log2FoldChange']
                        pdf_exon[,i] <- res[,'pvalue']
                        i <- i+1

                }
                for(a in intron_cons)
                {
                        contrast_intron <- numeric(nci)
                        contrast_intron[a] <- 1
                        print(intron_ratnames[a])
                        cname <- sub('Condition_', '', intron_ratnames[a])
			print(cname)
                        res <- as.data.frame( results(dds_intron, contrast = contrast_intron) )
                        rownames(log2df_intron) <- rownames(res)
                        rownames(pdf_intron) <- rownames(res)
                        colnames(log2df_intron)[j] <- cname
                        colnames(pdf_intron)[j] <- cname
                        log2df_intron[,j] <- res[,'log2FoldChange']
                        pdf_intron[,j] <- res[,'pvalue']
                        j <- j+1
                }
	}
	write.table(log2df_exon, paste(outname,"_normalized.",normalization,".exonic.",method,".log2.txt",sep=""),quote=F, sep="\t")
	write.table(pdf_exon, paste(outname,"_normalized.",normalization,".exonic.",method,".pvalue.txt",sep=""),quote=F, sep="\t")
	write.table(log2df_intron, paste(outname,"_normalized.",normalization,".intronic.",method,".log2.txt",sep=""),quote=F, sep="\t")
        write.table(pdf_intron, paste(outname,"_normalized.",normalization,".intronic.",method,".pvalue.txt",sep=""),quote=F, sep="\t")
}






# Generate design information from sample table
design <- as.data.frame(sampleTable)
# Design info only needs everything from one data type
design <- design[design$ReadType=='exonic',]
# Desing info only contains sample names and the condition, or cell type
if( normalization == 'Batch' || normalization == 'BatchCondition') {
	design <- design[,c('Samples', 'Batch', 'Condition')]
} else {
	design <- design[,c('Samples', 'Condition')]
}
rownames( design ) <- design[,1]

print('Diffrac Design')
print(design)

# Run DiffRAC
if( normalization == 'Batch' || normalization == 'BatchCondition') {
	diffrac_res <- DiffRAC( ~ Condition + Batch,
        design,
        exon_counts,intron_counts,
        "condition",bias = 1)        
} else {
      	diffrac_res <- DiffRAC( ~ Condition,
        design,
        exon_counts,intron_counts,
        "condition",bias = 1)
}

#The design data frame:
dmat <- as.data.frame( diffrac_res$model_mat )
print(dmat)

# Number of columns in dmat
nc <- ncol(dmat)
# Column names
ratnames <- colnames(dmat)
# Find indexes of columns with 'Ratio' in them
allratio <- grep('(?=.*Condition)(?=.*Ratio)', ratnames, perl = T)

print(ratnames[allratio])
print(allratio)
nallratio <- length(allratio)

i <- 1
if(method == 'Mean') # Mean method return differntial expression to the mean of all conditions
{
        log2df <- data.frame(matrix(NA, nrow = nrow(diffrac_res$dds), ncol = nallratio+1)) # initialize empty data frames
	pdf <- data.frame(matrix(NA, nrow = nrow(diffrac_res$dds), ncol = nallratio+1))
	contrast <- numeric(nc) # contrast method can return differential expression for combination of colums that are in the design matrix
	# For example with three conditions, first condition is the control:
	# then the last three lines represent
	# Nominator = ln2(KDcontrol)
	# 1:Ratio = ln2(KD1) - ln2(KDcontrol)
	# 2:Ratio = ln2(KD2) - ln2(KDcontrol)
	# >> ln2(KD1) = 1:Ratio + ln(KDcontrol) = 1:Ratio + Nominator
	# Mean = (Nominator*3 + 1:Ratio + 2:Ratio)/3
	# Or in general for N conditions: Nominator + (1:Ratio + 2:Ratio + 3 :Ratio + ...)/N
	# Then the difference to the mean is: ln2(KD1) - ln2(KDmean) = 1:Ratio - (1:Ratio + 2:Ratio)/3
	# To get this, we set contrast = c(0,0,0,0,2/3,-1/3)

	# We start with ln(KDcontrol) - ln(KDmean)
        contrast[allratio] <- -1/(nallratio+1)
        res <- as.data.frame( results(diffrac_res$dds, contrast = contrast) )
        colnames(log2df)[i] <- 'Control_to_MEAN'
        colnames(pdf)[i] <- 'Control_to_MEAN'
        log2df[,i] <- res[,'log2FoldChange']
        pdf[,i] <- res[,'pvalue']
        i <- i+1
	for(a in allratio)
        {
                contrast <- numeric(nc)
		contrast[allratio] <- -1/(nallratio+1)
                contrast[a] <- contrast[a] + 1
                print(ratnames[a])
		cname <- paste(sub('Condition', '', sub(":Ratio", '', ratnames[a])), 'to_MEAN', sep = '')
                print(contrast)
                res <- as.data.frame( results(diffrac_res$dds, contrast = contrast) )
		rownames(log2df) <- rownames(res)
		rownames(pdf) <- rownames(res)
		colnames(log2df)[i] <- cname
                colnames(pdf)[i] <- cname
		log2df[,i] <- res[,'log2FoldChange']
		pdf[,i] <- res[,'pvalue']
		i <- i+1

        }

} else
{
	log2df <- data.frame(matrix(NA, nrow = nrow(diffrac_res$dds), ncol = nallratio))
	pdf <- data.frame(matrix(NA, nrow = nrow(diffrac_res$dds), ncol = nallratio))
	print('Data frames initializes')
	print(ncol(pdf))
	print(nrow(pdf))
	for(a in allratio)
	{
		contrast <- numeric(nc)
		# if we just want to iterate over all ratios that were computed, we just have to use a vector that has all 0's but a 1 at the column of interest
		contrast[a] <- 1
		print(ratnames[a])
		cname <- paste(sub('Condition', '', sub(":Ratio", '', ratnames[a])), 'to_CTRL', sep ='')
		print(contrast)
		res <- as.data.frame( results(diffrac_res$dds, contrast = contrast) )
		rownames(log2df) <- rownames(res)
                rownames(pdf) <- rownames(res)
                colnames(log2df)[i] <- cname
                colnames(pdf)[i] <- cname
                log2df[,i] <- res[,'log2FoldChange']
                pdf[,i] <- res[,'pvalue']
                i <- i+1
		
	}

}


write.table(log2df, paste(outname,"_diffrac.",normalization,'.',method,".log2foldchange.txt",sep=""),quote=F, sep="\t")
write.table(pdf, paste(outname,"_diffrac.",normalization,'.',method,".pvalue.txt",sep=""),quote=F, sep="\t")


