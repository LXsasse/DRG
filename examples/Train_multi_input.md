# Train CNNs with multiple sequence inputs

## Define data
```
input=TSS40k.npz # DNA input around the TSS
trinput=RNA40k.npz # Transcript input (for model with post-transcriptional understanding)

output0=exonic.tsv # Gene expression counts for all cell types and interleukins
troutput0=intronic.tsv # Transcription rate counts for DNA sequence for all cell types and interleukins
deoutput0=degrad.tsv # Degradation rate counts for transcript forall cell types and interleukins
```

## Other files for training and performance testing

All models are tested on how well they can predict correlation across interleukins for each gene in a cell type specific way.
```
classfile=condition.class.txt # a class table assigns classes, here Cell types for all conditions, to the tracks in the csv files.

cv=Exintron_testsetcv10.txt # Test and training fold that leave chromosomes out for test and validation
fold=1

scriptdir=/path/to/DRG/scripts/train_models/
```

## Define model architectures

```
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
```

## Define optimization parameters

```
bs=8 # batchsize
pat=10 # patience
lr=0.00001

opt=SGD+optim_params=0.9 #$1 # 'AdamW' 'AdamW+optim_weight_decay=0.1' 'Adam' 'SGD+optim_params=0.9' # Try different optimizers instead of different seeds

device='cuda:0'
seed=1
outdir=Models/

training=conv_batch_norm=True+fc_dropout=0.1+lr=${lr}+patience=${pat}+batchsize=${bs}+optimizer=${opt}+device=${device}+keepmodel=True+seed=${seed}+finetuning_patience=3+finetuning_rounds=2+finetuning_rate=0.1+init_adjust=False+warm_up_epochs=4

addname='sd'${seed} # added to final file name
```

## A) Baseline single sequence model: DNA input -> exonic counts
Trained with Correlation + MSE loss
```
trloss=Correlationmse
python ${scriptdir}run_cnn_model.py $trinput $output0 --outdir $outdir --delimiter $'\t' --reverse_complement --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss=Correlationdata+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --addname $addname 
```

## B) Single input multi-modal output model: DNA input -> exonic,degradation,transcription counts
Trained on same loss but validation loss is only looking at expression: "['Correlationdata','None','None']"
```
python ${scriptdir}run_cnn_model.py $input ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname
```

## C) Multi-sequence input single output model with NN combinations: DNA + RNA input -> Expression
```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss=Correlationdata+${basemodel}+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --addname $addname
```

## D) Multi-input mulit-output model with "real combination": DNA --> KT, DNA + RNA --> Expression, RNA --> KD
BUT indivudal linear embedding for each data modality (shared_embedding=False)
```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass="['difference','direct','direct']"+combine_function=Linear+${training}+shared_embedding=False --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname
```
If the prediction of KT and degradation should also be used for the difference for Expression, then set shared_embedding=True
```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass="['difference','direct','direct']"+combine_function=Linear+shared_embedding=True+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname
```

## F) Multi-input multi-output model with independent combinations through fully connected net: DNA + RNA --> KT, DNA + RNA --> Expression, RNA +RNA --> KD

Both inputs are used to predict each data modality with 2 fully connected layers for each modality that mix the information from each sequence embedding

```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+n_combine_layers=2+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname
```

Instead, we can also define that only the expression modality sees both embeddings. 
```
python ${scriptdir}run_cnn_model_multi.py ${input},${trinput} ${output0},${deoutput0},${troutput0} --outdir $outdir --delimiter $'\t' --reverse_complement True,False --crossvalidation $cv $fold 10 --cnn loss_function=${trloss}+validation_loss="['Correlationdata','None','None']"+${basemodel}+input_to_combinefunc='[[0,1],[1],[0]]'+outclass=Linear+combine_function=GELU+n_combine_layers=2+${training} --split_outclasses $classfile --save_correlation_perclass --save_correlation_perpoint --add_fileclasses ex,kd,kt --addname $addname
```
