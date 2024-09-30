# Train CNN with single input sequence and single output matrix

## Define sequence and data matrix
```
inseq=../data/Testin.npz
target=../data/Test.csv # Data matrix with data points in rows, and tracks in columns
crossval_file=10 # Can either be number of folds, or a file with # Set N, followed by a line with all data point names that are in this set. 
```
## Define model architecture
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
splitout=../data/Testclasses.txt # Assign classes to track in target matrix to compute correlation across cells for each class individually for example.

python ../scripts/train_models/run_cnn_model.py $inseq $target --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint --save_correlation_perclass 
````

## Train model on subset of selected tracks  
```
selected_tracks=B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp,20,33,54 # this can be the names in the header of $target or indeces
python ../scripts/train_models/run_cnn_model.py $inseq $target --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint --select_tracks # selected_tracks
```


## Train model with two data modality matrices
Now, we can either just have individual linear prediction heads, but add individual training and validation losses
Or we even use individual fully connected layers for each modality

```
target2=../data/Test2.csv

loss_function=Correlationmse
validation_loss="[Correlationdata,None]" # Second data modality is only included for training but the loss is not considered for validation and early stopping
nfc_layers=[2,2] #each modality gets its own 2 fully connected layers

cnn='l_kernels='${l_kernels}'+num_kernels='${num_kernels}'+pooling_size='${pooling_size}'+dilated_convolutions='${dilated_convolutions}'+fclayer_size='${fclayer_size}'+nfc_layers=${nfc_layer}
training='epochs=${epochs}+lr=${lr}+finetuning=${finetuning}+keepmodel=${keepmodel}+loss_function=${loss_function}+validation_loss=${validation_loss}'

python ../scripts/train_models/run_cnn_model.py $inseq ${target},${target2} --cross_validation $crossval_file 0 0 --cnn ${cnn}+${training} --split_outclasses $splitout --save_correlation_perpoint
```

