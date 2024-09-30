# Load models to make predictions on selected tracks

```
inseq=../data/Testin.npz
target=../data/Test.csv # Data matrix with data points in rows, and tracks in columns
crossval_file=10 # Can either be number of folds, or a file with # Set N, followed by a line with all data point names that are in this set.
splitout=../data/Testclasses.txt # Assign classes to track in target matrix to compute correlation across cells for each class individually for example.
```
## Single modality mulit-task model
```
# Single modality, multi-task model arguments and parameters
paramssh=TestonTestin-cv10-0_MSEk100l7TfGELUmax5_dc2i1d1s1l7da4r1nfc2s512tr1e-05Adam-F_model_params.dat

#python ../scripts/train_models/run_cnn_model.py $inseq $target --cross_validation $crossval_file 0 0 --cnn $paramssh --split_outclasses $splitout --save_correlation_perpoint --save_predictions
# --save_predictions saves all predicted values for data points in the test set across tracks in npz format
```

## Mulit-modality mulit-task model

```
# Multi-modality, multi-task model arguments and parameters
paramsmh=TestTest2onTestin-cv10-0_MSEk100l7TfGELUmax5_dc2i1d1s1l7da4r1nfc2s512tr1e-05Adam-F_model_params.dat
target2=../data/Test2.csv

python ../scripts/train_models/run_cnn_model.py $inseq ${target},${target2} --cross_validation $crossval_file 0 0 --cnn $paramsmh --split_outclasses $splitout --save_correlation_perpoint --save_predictions

```
