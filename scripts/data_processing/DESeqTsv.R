library(DESeq2)
#commandArgs picks up the variables you pass from the command line
args <- commandArgs(trailingOnly = TRUE)

tsv <- args[1] # count file
pasAnno <- args[2] # info file
outname <- args[3] # name for output file
method <- args[4] # method to use for differential expression analysis: compare to mean, or name 

cts <- as.matrix(read.csv(tsv, sep="\t", row.names = 1, check.names =FALSE, comment.char=""))
coldata <- read.csv(pasAnno, row.names=1, sep = '\t', check.names = FALSE, header = FALSE)
#head(coldata)
all(rownames(coldata) %in% colnames(cts))
cts <- cts[, rownames(coldata)]
all(rownames(coldata) == colnames(cts))
print(colnames(coldata))

### Not sure if we need design ~1 or if we can use the design matrix with cell types to get the mean counts for each cell type with counts() and with vst(). Does counts and vst use the design matrix?
dds <- DESeqDataSetFromMatrix(countData = cts, colData = coldata, design = ~ 1)

print('Design')
print(design(dds))

# Get normalized counts
dds <- DESeq(dds)
ndds <- counts(dds, normalized = TRUE)
write.table(ndds, paste(outname,"_normed.txt", sep=""),quote=F, sep="\t")

# Get variance stabilzed counts
vsd <- vst(object = dds, blind = T)
write.table(assay(vsd), paste(outname,"_vst.txt",sep=""),quote=F, sep="\t")

# Get fold changes to mean
# change the design choices for DE analysis
### if method not mean, I would like method to define the column that we should compare everything else against
colnames(coldata) <- "sample"
dds <- DESeqDataSetFromMatrix(countData = cts, colData = coldata, design = ~ V2)
print(design(dds))
if(method != "Mean") {
dds$V2 <- relevel(dds$V2, ref = method)
}
### Sometimes takes for ever and doesn't converge for days, does not give an error message, why?
dds <- DESeq(dds)
ratnames <- resultsNames(dds)
print(ratnames)
# grep everything but the intercept
cons <- grep("V2", ratnames)

# number of all conditions, including the one that is used as control
nce <- length(ratnames)
i <- 1
if(method == 'Mean') {
	log2df <- data.frame(matrix(NA, nrow = nrow(dds), ncol = nce+1)) # initialize empty data frames
	pdf <- data.frame(matrix(NA, nrow = nrow(dds), ncol = nce+1))
	contrast <- numeric(nce) # contrast method can return differential expression for combination of colums that are in the design matrix
	# here we compare the column that was used as control to the mean
	contrast[cons] <- -1/nce
	contrast[1] <- 0
	
	### This may all be wrong: Contrast might just provide the coefficients to sum up the columns in the ratnames. In this case, the first entry for the intercept should be 0, and the other ones should be -1/nce, and further below they should be -1/nce and the one at 'a' should be 1-1/nce.
	# Note, if these are just the coefficients then the mean is not the arithmetic mean, but the geometric mean
	### Nevertheless, whichever is correct. It is very confusing that this takes so long. It seems like DESeq is recomputing certain statistics for the fold changes rather than just returning the sum of pre-calculated log2 fold changes of other combinations. 
	# Is there a way to get the whole matrix at once, so it's quick? --> even if we have to add a pseudo mean. 
	res <- results(dds, contrast = contrast)
	res <- as.data.frame(res)
	cname <- paste(sapply(strsplit(ratnames[2], '_vs_'), '[', 2),'_vs_Mean',sep='')
	print(cname)
	colnames(log2df)[i] <- cname
	colnames(pdf)[i] <- cname
	log2df[,i] <- res[,'log2FoldChange']
	pdf[,i] <- res[,'pvalue']
	i <- i+1
	for(a in cons) {
		contrast <- numeric(nce)
		contrast[cons] <- -1/(nce)
		contrast[a] <- contrast[a] + 1
		contrast[1] <- 0
		cname <- paste(sapply(strsplit(sapply(strsplit(ratnames[a], '_vs_'), '[', 1), 'V2_'), '[',2),'_vs_Mean',sep='')
		print(cname)
		#res <- results(dds, contrast = contrast)
		res <- lfcShrink(dds, contrast = contrast, type="ashr")
		### instead of results, I would like to shrink the fold changes but I don't know how to do that for comparison to the mean with contrast
		# There is an idea here: https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#extended-section-on-shrinkage-estimators
		#res <- lfcShrink(res, coef = a, type="apeglm")
		res <- as.data.frame(res)
		rownames(log2df) <- rownames(res)
		rownames(pdf) <- rownames(res)
		colnames(log2df)[i] <- cname
		colnames(pdf)[i] <- cname
		log2df[,i] <- res[,'log2FoldChange']
		pdf[,i] <- res[,'pvalue']
		i <- i+1
		}
} else {
	# Compare all samples to one that was selected as control
	log2df <- data.frame(matrix(NA, nrow = nrow(dds), ncol = nce)) # initialize empty data frames
	pdf <- data.frame(matrix(NA, nrow = nrow(dds), ncol = nce))

	i <- 1
	for(a in cons) {
		contrast <- numeric(nce)
		contrast[a] <- 1
		print(ratnames[a])
		cname <- sub('V2_', '', ratnames[a])
		print(cname)
		#res <- results(dds, contrast = contrast)
		# We use shrunken log fold changes here instead of the default fc from the results. Shrunken log fold changes are apparently standard for vulcano plots
		
		res <- as.data.frame(res)
		rownames(log2df) <- rownames(res)
		rownames(pdf) <- rownames(res)
		colnames(log2df)[i] <- cname
		colnames(pdf)[i] <- cname
		log2df[,i] <- res[,'log2FoldChange']
		pdf[,i] <- res[,'pvalue']
		i <- i+1
		}
}
write.table(log2df, paste(outname,'.',method,".log2.txt",sep=""),quote=F, sep="\t")
write.table(pdf, paste(outname,".",method,".pvalue.txt",sep=""),quote=F, sep="\t")


					
					
					
	
                
