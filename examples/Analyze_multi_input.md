# Compute kernel effects and sequence attributions for multi-sequence input

## Define all necessary data and arguments

```
# DNA input (for base model)
input=TSS40k.npz # TSS100k.npz
# Transcript input (for model with post-transcriptional understanding)
trinput=RNA40k.npz # RNA100k.npz
#trinput=RNA40kc.npz

# Gene expression counts for all cell types and interleukins
output0=exonic.tsv

# Transcription rate counts for DNA sequence for all cell types and interleukins
troutput0=intronic.tsv

# Degradation rate counts for transcript forall cell types and interleukins
deoutput0=degrad.tsv

### All models are tested on how well they can predict correlation across interleukins for each gene in a cell type specific way.
# a class table assigns the classes to the different columns in exonic.counts.txt
classfile=condition.class.txt

cv=Exintron_testsetcv10.txt # Test and training fold that leave chromosomes out for test and validation
fold=1

scriptdir=~/Scripts/Git/DRG/scripts/train_models/

nk=300 # number of kernels
lk=15 # kernel length

fps=4 # initial pooling size after first layer

ndp=3 # number of dilated convolutions without pooling but residuals and dilations. 
dil='[1,2,4]' # dilations in ndp layers
lkc=11 # kernel size in conv blocks

ps=6 # pooling size of transformer convs for 40k
#ps=7 # pooling size of transformer convs for 100k
dc=4 # number of tranfomer_convolutions that reduce the length of the sequence with pooling

fcls=1024 # size of flattened input to fully connected layers
nfc=3 # Number of fully connected layers

# Model with dilated convolutions and weighted mean pooling in subsequent layers
basemodel=num_kernels=${nk}+l_kernels=${lk}+max_pooling=False+weighted_pooling=True+pooling_size=${fps}+net_function=GELU+dilated_convolutions=${ndp}+l_dilkernels=${lkc}+dilations=${dil}+transformer_convolutions=${dc}+l_trkernels=${lkc}+trweighted_pooling=${ps}+fclayer_size=${fcls}+nfc_layers=${nfc}

# Training parameter
bs=8 # batchsize
pat=10 # patience
lr=0.00001

opt=SGD+optim_params=0.9 #$1 # 'AdamW' 'AdamW+optim_weight_decay=0.1' 'Adam' 'SGD+optim_params=0.9' # Try different optimizers instead of different seeds

device='cuda:0'
seed=1
outdir=Models/
```

## Load model parameters to save pwms and kernel importance and effects

```
tset='Testset1.txt' # Test set sequences that are used to analyze the models learned grammar
# Alternatively, could be filtered for data points for which the model makes good predictions
# File that saves arguments to initialize models *model_params.dat
# Parameters are saved in *parameter.pth files
parms=${outdir}exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_model_params.dat

python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --add_fileclasses ex,kd,kt --split_outclasses $classfile --kernel_analysis --save_kernel_filters --save_correlation_perpoint
```

## Load model parameters to save TISMs/corrected gradients, deepshap does not work for multi-sequence models yet
When one computes the gradients for a sequence, one can define for which tracks these attributions should be computed. Since each attribution is computed for all bases, these files can get very large. Therefore, we select tracks but also compute these only for a very limited number sequences. Moreover, we only extract the 500 best attributions along the sequence. To identify motifs, we add another channel that saves the position of the attributions in the attributions.
```
tset=<genename1,genename2>
seltracks=Bfo_PBS_ex,DC8+_PBS_ex,MC_PBS_ex,MFRP_PBS_ex,MF_PBS_ex,MZB_PBS_ex,Mo6C+_PBS_ex,NK_PBS_ex,T4_PBS_ex,T8_PBS_ex,Tgd_PBS_ex,Treg_PBS_ex,pDC_PBS_ex
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --add_fileclasses ex,kd,kt --sequence_attributions grad $seltracks --topattributions 500 --seqattribution_name PBSs2

# --topattributions only saves the 500 positions of the strongest attributions in each celltype, the shape of the base-type channel is extended by one with the position of the attributions in sequence, from 4 bases (ACGT) to 5 (ACGT +position)
# --sequence_attributions <type> <tracks> 
# <type> can be grad, deepshap, ism, and deeplift: ism takes too long and deepshap and deeplift do not work for these models yet
# <tracks> can be all, integers separated by ',', or track headers that should be selected
```

Instead of selecting a subset of cell types, we can also average cell types for attributions. This average can also be across all conditions for a data type, f.e. exonic, intronic, and degradation. However, if the names of the cell types are already data type specific in the output files, f.e. B.fo_kd, B.fo_ex, then the $classfile should also be data type specific, i.e B.fo_kd => B_kd. Otherwise if the the different output files have overlapping cell types names in them, the classes can be defined independent of the data type, e.g. B.fo => B 
```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --add_fileclasses ex,kd,kt --average_outclasses $classfile --sequence_attributions grad all --topattributions 500 --seqattribution_name avg
```



