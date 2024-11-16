# Train CNN with single sequence input, single output, or multiple output matrices.

## Define sequence and data matrix

```
inseq=../data/Testin.npz # one hot encoded sequences (see processing for content of file)
target=../data/Test.csv # Data matrix with data points in rows, and tracks in columns
crossval_file=10 # Can either be number of folds, or a file with # Set N \n, followed by a line with all data point names that are in this set separated by a space. 
```
## Define the model architecture

Please see drg_tools/cnn_model.py for additional arguments and possible model architectures

```
l_kernels=7 # kernel size
num_kernels=100 # Number of kernels in first layer
pooling_size=5 # pooling size after first layer
dilated_convolutions=2 # number of conv. blocks
dilmax_pooling=4 # max pooling size after each conv. block
fclayer_size=512 # dimension for fully connected layer
nfc_layers=2 # number of fully connected layers after conv. blocks
epochs=10 # training epochs
finetuning=False # Fine tuning with reduced lr
keepmodel=True # Keep model parameters after training and computation of performance etc. 
lr=0.00001 # learning rate

# Combine individual choices to provide for the model
cnn='l_kernels='${l_kernels}'+num_kernels='${num_kernels}'+pooling_size='${pooling_size}'+dilated_convolutions='${dilated_convolutions}'+fclayer_size='${fclayer_size}'+nfc_layers=${nfc_layer}
training='epochs=${epochs}+lr=${lr}+finetuning=${finetuning}+keepmodel=${keepmodel}'
```

## Train model

Train model, compute correlations for tracks (--save_correlation_perclass) and data points in test set (--save_correlation_perpoint)
```
splitout=../data/Testclasses.txt # Assign classes to tracks in target matrix to compute correlation across cells for each class individually. Otherwise computes correlation across all tracks.

python ../scripts/train_models/run_cnn_model.py $inseq $target --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint --save_correlation_perclass 
````

## Train model on subset of selected tracks  
Instead of training on all tracks in the data file, one can also select columns based on their names or indexes. Of course, this can be done independent of the modeling beforehand. 

```
selected_tracks=B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp,20,33,54 # this can be the names in the header of $target or indeces
python ../scripts/train_models/run_cnn_model.py $inseq $target --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint --select_tracks # selected_tracks
```

## Train model with multiple data modality matrix

When we have multiple data matrices on which we can train the model, e.g. different modalities, classical approaches just concacetenated these data sets and had linear prediction heads differentiate between them. However, different modalities may not be linearly connected, so that we either want to provide individual non-linear prediction heads or different loss functions to the individual matrices.

```
target2=../data/Test2.csv

loss_function=Correlationmse  # Loss function is applied to every modality individually, that means that correlations will be computed within a modality. 
validation_loss="[Correlationdata,None]" # Second data modality is only included for training but the loss is not considered for validation and early stopping
nfc_layers=[2,2] #each modality gets its own 2 fully connected layers

cnn='l_kernels='${l_kernels}'+num_kernels='${num_kernels}'+pooling_size='${pooling_size}'+dilated_convolutions='${dilated_convolutions}'+fclayer_size='${fclayer_size}'+nfc_layers=${nfc_layer}
training='epochs=${epochs}+lr=${lr}+finetuning=${finetuning}+keepmodel=${keepmodel}+loss_function=${loss_function}+validation_loss=${validation_loss}'

python ../scripts/train_models/run_cnn_model.py $inseq ${target},${target2} --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint
```

