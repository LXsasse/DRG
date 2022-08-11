This directory contains cnn modules (i.e. cnn_model.py, bpcnn_model.py, cnn_model_multi.py) and its submodules. It also contains functions to train modules, compute and save performance, and read and process input data. 

	cnn_model.py:
		Multi-task CNN which uses one or several one-hot encoded sequences of same length as input and predicts single outputs for N tracks for these sequences

	cnn_model_multi.py:
		Multi-task CNN which uses several one-hot encoded sequences of same length as input, creates individual CNNs for each input, combines there with fully connected layers, and predicts single outputs for N tracks for each sequence.

	bpcnn_model.py: 
		Multi-task CNN which uses one or several one-hot encoded sequences of same length as input and predicts M bined outputs for N tracks for these sequences.
		The main difference to the above models is that bpcnn outputs a 3D tensor (N_seqs, N_tracks, M_bins). It achieves this by using a convolutional layer as a final linear layer instead of a fully connected layer.
		
		Run:
		$python bpcnn_model.py <one-hot_input.npz> <bp_binned_output_tracks.npz> --outdir preferred_dir/ --crossvalidation cv_file.txt n_fold N_folds --cnn loss_function=MSE+num_kernels=100+l_kernels=25+dilated_convolutions=2+conv_increase=1.+dilations=[2,4]+l_dilkernels=4+dil_residual_entire=True+batchsize=10+write_steps=1+lr=0.001+restart=True+keepmodel=True --save_jensenshannon_pergene --save_msemean_pergene --save_mse_pergene --save_msewindow_pergene --msemeanaverage
		
		Look at bpcnn module for description of individual parameter of the module
		
		Load model parameters and make predictions for unseen sequences:
		
		$python bpcnn_model.py <one-hot_input.npz> <None or matching_outputtotestperformance.npz> --outdir preferred_dir/ --predictnew --cnn cnn_outputfile_model_params.dat --save_predictions

# This is a pull request test
