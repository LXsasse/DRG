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

# Could also run with combined model split across all cell types

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

# One attention layer, 4 heads, after dilated convolutions before pooling with conv. blocks
transmodel=num_kernels=${nk}+l_kernels=${lk}+max_pooling=False+weighted_pooling=True+pooling_size=${fps}+net_function=GELU+dilated_convolutions=${ndp}+l_dilkernels=${lkc}+dilations=${dil}+dilweighted_pooling=10+dilpooling_steps=3+n_attention=1+n_distattention=4+dim_distattention=1.8+transformer_convolutions=${dc}+l_trkernels=${lkc}+trweighted_pooling=4+fclayer_size=${fcls}+nfc_layers=${nfc}


bs=8 # batchsize
pat=10 # patience
lr=0.00001

opt=SGD+optim_params=0.9 #$1 # 'AdamW' 'AdamW+optim_weight_decay=0.1' 'Adam' 'SGD+optim_params=0.9' # Try different optimizers instead of different seeds

device='cuda:0'
seed=1
outdir=Models/

training=conv_batch_norm=True+fc_dropout=0.1+lr=${lr}+patience=${pat}+batchsize=${bs}+optimizer=${opt}+device=${device}+keepmodel=True+seed=${seed}+finetuning_patience=3+finetuning_rounds=2+finetuning_rate=0.1+init_adjust=False+warm_up_epochs=4

addname='sd'${seed}

# A) Baseline model trained on Correlation + MSE loss: DNA input, exonic counts output0
trloss=Correlationmse
python ${scriptdir}run_cnn_model.py $trinput $output0 --outdir $outdir --delimiter $'\t' --reverse_complement --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss=Correlationdata+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --addname $addname 

# B) Baseline single input multi-task output model: DNA input, exonic counts validation loss: ${output0},${deoutput0},${troutput0}, "['Correlationdata','None','None']"
python ${scriptdir}run_cnn_model.py $input ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname

# C) Multi-input single output model with NN combinations: DNA and RNA input --> Correlation + MSE loss: DNA + RNA --> Expression
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss=Correlationdata+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --addname $addname

# D) Multi-input-output model with real combination BUT linear embedding for each data modality: DNA and RNA input --> Correlation + MSE loss: DNA --> KT, DNA + RNA --> Expression, RNA --> KD with realistic combination
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass="['difference','direct','direct']"+combine_function=Linear+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname

# E) Multi-input-output model with real combinations: DNA and RNA input --> Correlation + MSE loss: DNA --> KT, DNA + RNA --> Expression, RNA --> KD with realistic combination (hard combination with shared outputs)
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass="['difference','direct','direct']"+combine_function=Linear+shared_embedding=True+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname

# F) Multi-input-output model with independent combinations through fully connected net: DNA and RNA input --> Correlation + MSE loss: DNA --> KT, DNA + RNA --> Expression, RNA --> KD with realistic combination
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+n_combine_layers=2+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname

# F) Multi-input-output attention model with real combinations and 100k input: DNA and RNA input --> Correlation + MSE loss: DNA --> KT, DNA + RNA --> Expression, RNA --> KD with realistic combination
#input=TSS100k.npz
#trinput=RNA100k.npz
#python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${transmodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass="['difference','direct','direct']"+combine_function=Linear+${training}+batchsize=2 --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname

#device='cuda:0'
#tset='Testset1.txt'
#classfile=celltype.class.txt
#tset='Testsingle.txt'
#parms=${outdir}exonicdegradintroniconTSS40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4dNoned1s1r1l11mw6nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs8-F_model_params.dat
#python ${homedir}/Scripts/DRG/cnn_model.py $input ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device} --split_outclasses $classfile -convertedkernel_ppms -save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt 
#parms=${outdir}exoniconTSS40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCota_dc3i1d1-2-4s1l11r1_tc4dNoned1s1r1l11mw6nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs8-F_model_params.dat
#python ${homedir}/Scripts/DRG/cnn_model.py $trinput $output0 --outdir $outdir --delimiter $'\t' --reverse_complement --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device} --split_outclasses $classfile --convertedkernel_ppms --onlyppms -save_correlation_perclass -save_correlation_perpoint 
#parms=${outdir}exoniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCota_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb512nl3GELU1.1r0_model_params.dat
#python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --split_outclasses $classfile --add_fileclasses ex,kd,kt -convertedkernel_ppms -save_correlation_perclass --save_correlation_perpoint 
#parms=${outdir}exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024LiarcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb512nl0GELU1.1r0_model_params.dat
#parms=${outdir}exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_combse88nl0Linear1.1r0_model_params.dat
#time python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device} --split_outclasses $classfile --add_fileclasses ex,kd,kt --convertedkernel_ppms --onlyppms -genewise_kernel_impact 0.7 -save_correlation_perclass -save_correlation_perpoint
# Load model parameters to save pwms and kernel importance
#parms=${outdir}exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_model_params.dat
#time python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --split_outclasses $classfile --add_fileclasses ex,kd,kt --convertedkernel_ppms --onlyppms -genewise_kernel_impact 0.7 -save_correlation_perclass -save_correlation_perpoint 

parms=${outdir}exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_model_params.dat
#tset='Testset1.txt'
#tset='Testsingle.txt'
#tset=exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclPBSex_Max4.6_MaxDiff1.0_00.5sim_list.txt
tset=pbslist.txt
# Load model parameters to save TISMs
time python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --split_outclasses $classfile --add_fileclasses ex,kd,kt --grad Bfo_PBS_ex,DC8+_PBS_ex,MC_PBS_ex,MFRP_PBS_ex,MF_PBS_ex,MZB_PBS_ex,Mo6C+_PBS_ex,NK_PBS_ex,T4_PBS_ex,T8_PBS_ex,Tgd_PBS_ex,Treg_PBS_ex,pDC_PBS_ex --topattributions 500 --gradname PBSs2
exit
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclBfoexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclDC8exlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclMCexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclMFexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclMFRPexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclMo6Cexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclMZBexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclNKexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclpDCexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclT4exlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclT8exlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclTgdexlogp10_fc1.0_p1.301_00.5sim_list.txt
#tset=Models/exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclTregexlogp10_fc1.0_p1.301_00.5sim_list.txt
#grads=Bfo_IL15_ex,Bfo_IL2_ex,Bfo_IL21_ex,Bfo_IL4_ex,Bfo_IL7_ex,Bfo_IL9_ex,Bfo_PBS_ex
#grads=DC8+_IL15_ex,DC8+_IL2_ex,DC8+_IL21_ex,DC8+_IL4_ex,DC8+_IL7_ex,DC8+_IL9_ex,DC8+_PBS_ex
#grads=MC_IL15_ex,MC_IL2_ex,MC_IL21_ex,MC_IL4_ex,MC_IL7_ex,MC_IL9_ex,MC_PBS_ex
#grads=MF_IL15_ex,MF_IL2_ex,MF_IL21_ex,MF_IL4_ex,MF_IL7_ex,MF_IL9_ex,MF_PBS_ex
#grads=MFRP_IL15_ex,MFRP_IL21_ex,MFRP_IL4_ex,MFRP_IL7_ex,MFRP_PBS_ex
#grads=Mo6C+_IL15_ex,Mo6C+_IL2_ex,Mo6C+_IL21_ex,Mo6C+_IL4_ex,Mo6C+_IL7_ex,Mo6C+_IL9_ex,Mo6C+_PBS_ex
#grads=MZB_IL15_ex,MZB_IL2_ex,MZB_IL21_ex,MZB_IL4_ex,MZB_IL7_ex,MZB_IL9_ex,MZB_PBS_ex
#grads=NK_IL15_ex,NK_IL2_ex,NK_IL21_ex,NK_IL4_ex,NK_IL7_ex,NK_PBS_ex
#grads=pDC_IL15_ex,pDC_IL2_ex,pDC_IL21_ex,pDC_IL4_ex,pDC_IL7_ex,pDC_IL9_ex,pDC_PBS_ex
#grads=T4_IL15_ex,T4_IL2_ex,T4_IL21_ex,T4_IL4_ex,T4_IL7_ex,T4_IL9_ex,T4_PBS_ex
#grads=T8_IL15_ex,T8_IL2_ex,T8_IL21_ex,T8_IL4_ex,T8_IL7_ex,T8_IL9_ex,T8_PBS_ex
#grads=Tgd_IL15_ex,Tgd_IL2_ex,Tgd_IL21_ex,Tgd_IL4_ex,Tgd_IL7_ex,Tgd_IL9_ex,Tgd_PBS_ex
#grads=Treg_IL15_ex,Treg_IL2_ex,Treg_IL21_ex,Treg_IL4_ex,Treg_IL7_ex,Treg_IL9_ex,Treg_PBS_ex
#time python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device}+shared_embedding=False --split_outclasses $classfile --add_fileclasses ex,kd,kt --grad $grads --topattributions 500 --gradname TREG

exit

#parms=${outdir}exoniconTSS40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCota_dc3i1d1-2-4s1l11r1_tc4dNoned1s1r1l11mw6nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs8-F_model_params.dat
#time python ${homedir}/Scripts/DRG/cnn_model.py ${input} ${output0} --delimiter $'\t' --reverse_complement True --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}2set+device=${device} --split_outclasses $classfile --grad all --deeplift all -topattributions 250

# Load model parameters to save predictions and MSE loss
#time python ${homedir}/Scripts/DRG/cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --delimiter $'\t' --reverse_complement True,False --predictnew --select_list $tset --cnn ${parms} outname=${parms%_model_params.dat}+device=${device} --split_outclasses $classfile --add_fileclasses ex,kd,kt --save_predictions --save_mse_perpoint

