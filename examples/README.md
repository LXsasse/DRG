# Examples
This directory contains markdown sites with descriptions and code to execute different analyses.

## [Process data for model training](https://github.com/LXsasse/DRG/examples/Process_data.md)

To prepare model training, we download the reference genome, extract sequences in fasta, and transform those into one-hot encodings that we save in npz format. A future version will be able to take bed files as input to the model.

The model expects a normalized count matrix as target. The user can define for himself how to normalize and process this data. 

For model training, we usually split the data into train, validation, and test sets based on chromosomes. 

## [Train single input models](https://github.com/LXsasse/DRG/examples/Train_single_input.md)

This model uses a single one-hot encoded sequence to predict a normalized count matrix whose rows represent data points and columns represent different tracks, or conditions. This model can also split into diffferent individual networks after an initially shared part to better represent individual data matrices from different modalities. 

Define the model architecture, the training parameters, and the training and valiation set. 

Save model performance and visualize to compare to other models or assess model predictions

If training seems to have failed, plot the loss curves to get an idea what could have gone wrong. 

## [Train multi-sequence input models](https://github.com/LXsasse/DRG/examples/Train_multi_input.md) 

This model uses multiple sequences as input to predict 
complex phenotypes, such as gene expression. Similar to the single input model, this model can also predict multiple modalities with their own networks. However, this model can also connect the representations from each network throgh analytical functions.


## [Train BP-resolution model](https://github.com/LXsasse/DRG/examples/Train_bp_model.md)

BP-resolution models basically make a prediction for every base-pair that are used as input to the model. Modeling these profiles can help the model to learn more about that data, especially motif scaling, that helps it to improve predictions for total counts in the data matrix. 
This code can be used train on simple base-pair distributions or counts. 

## [Load set of parameters and finetune models on other data](https://github.com/LXsasse/DRG/examples/Finetune_models.md)

1. Load model and fine-tune on selected tracks

2. Load parts of model and fine-tune on other data set


## [Load models for inference and in silico experiments](https://github.com/LXsasse/DRG/examples/Inference_with_models.md)

Reload model to assess performance

## [Load model to assess global kernel effects](https://github.com/LXsasse/DRG/examples/Kernel_analysis.md)

For some model architectures, the convolutional kernels in the first layer of the model directly represent the binding patterns of the regulatory factors. In this case, we can determine the effect of these kernels on model predictions across model tracks, similar to in silico knock outs. 

The scripts return different statistics that tell us about different aspects of the data

## [Load model to look at sequence attributions](https://github.com/LXsasse/DRG/examples/Attribution_analysis.md)

Sequence attributions determine the importance of every base in the input to the model's predictions. For a multi-task model, attributions can be generated for each task, resulting in a massive data set to explore systematically. 

Usually, we select individual data points based on their performance and signal and plot their attributions across different tracks to investigate differences in motif activity

If the number of tracks or data points is limited, we extract attribution seqlets through statistical testing, cluster them, and compare for additional analysis

## [Perform analysis for muliti-sequence models](https://github.com/LXsasse/DRG/examples/Analyze_multi_input.md)

All these analysis can be performed for multi-sequence input models as well. A few details need to be adjusted here. 
