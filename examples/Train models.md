

# Test regular model

```
python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.00001 --split_outclasses ../data/Testclasses.txt -save_correlation_perpoint
````
# Test selection of tracks with regular model
```
python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.00001 --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp
```
# Test two output model with 
```
python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv,../data/Test2.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.00001 --split_outclasses ../data/Testclasses.txt -save_correlation_perpoint

python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv,../data/Test2.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.00001 --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp
```
# Test loading models to make predictions on selected tracks

```
paramssh=TestonTestin-cv10-0_MSEk100l7TfGELUmax5_dc2i1d1s1l7da4r1nfc2s512tr1e-05Adam-F_model_params.dat
paramsmh=TestTest2onTestin-cv10-0_MSEk100l7TfGELUmax5_dc2i1d1s1l7da4r1nfc2s512tr1e-05Adam-F_model_params.dat

#python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv --cross_validation 10 0 0 --cnn $paramssh --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.S

#python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv,../data/Test2.csv --cross_validation 10 0 0 --cnn $paramsmh --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp
```

# Test loading models and refine on subset of trask

```
#python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.0001+warm_up=False+warm_up_epochs=0 --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.S --load_parameters ${paramssh%model_params.dat}parameter.pth None None False False --addname ft

python ../scripts/train_models/run_cnn_model.py ../data/Testin.npz ../data/Test.csv,../data/Test2.csv --cross_validation 10 0 0 --cnn l_kernels=7+num_kernels=100+pooling_size=5+dilated_convolutions=2+dilmax_pooling=4+fclayer_size=512+nfc_layers=2+epochs=10+finetuning=False+keepmodel=True+lr=0.0001+warm_up_epochs=0 --split_outclasses ../data/Testclasses.txt --save_correlation_perpoint --select_tracks B.Fem.Sp,B.Fo.Sp,B.FrE.BM,B.GC.CB.Sp,B.GC.CC.Sp,B.MZ.Sp --load_parameters ${paramsmh%model_params.dat}parameter.pth None None False False --addname ft
```

