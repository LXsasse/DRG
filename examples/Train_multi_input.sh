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

