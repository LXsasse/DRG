# DRG

This is a repository with code to develop Convolutional Neural Networks (CNNs) for genomic sequence-to-function models that predict molecular phenotypes from genomic sequence, such as gene expression or chromatin accessibility. 
There are also scripts to visualize, compare and assess model performance or to compute and visualize feature importance. 

## Summary

The cis-regulatory code engraves the details about when, where, and how much of all gene products are created. It enables multi-cellular organisms to create several cell types from a single genome by regulating transcription, processing, and degradation of gene products. Unsurprisingly, variants that disturb the information in cis-regulatory elements can result in various genetic diseases. However, measuring the impact of all cis-regulatory sequence variations to function at a cell type specific basis is combinatorially infeasible.
Deep sequence-to-function models learn the relationship between genomic sequence and genome-wide functional molecular measurements. Trained on data from multiple cell types, these models are capable of developing a foundational understanding of the cis-regulatory code for individual cell types. However, recent evaluations have determined that the learned grammar is far from perfect, and that they are missing effects from distal elements as well as common and rare variants [[1,2]](#1,#2).

### Multi-modal multi-species multi-cell type models

To improve the foundational knowledge of these models about gene regulation, three directions seem to be promising: 1) Improving cell type resolution with single cell data. 2) Inclusion of different data modalities of different scales that measure different aspects of the multi-layered gene-regulatory process into a single model 3) Inclusion of data from other species to increase the number of available data modalities from different cell types and even more the sequence variance from which the model can learn evolutionary conserved cis-regulatory elements. Incorporating these data types in a biologically meaningful way at cell type resolution is a major challenge to gain a more comprehensive view on gene expression regulation. 

### Generating Genomic sequences from Sequence-to-function models

Deep neural networks can learn the link between genomic sequence and functional molecular measurements in a cell type specific manner from large scale measurement of molecular phenotypes. Utilizing the power of these models enables us, not only to extract the learn regulatory features, but also to design new sequences with cell type specific regulatory functions which can be exploited in bioengineering or new therapeutics. To generate these sequences, accurate sequence-to-function models are combined with generative processes that exploit the knowledge of the model to generate artificial sequences with target specific functionalities. However, how to effectively generate sequences that are cell type specific with improved functionality compared to wild-type sequences is an open research question. 


### Learning complex long-range interactions, such as structure for post-transcriptional regulation 

Gene expression is a multi-layered process and current models mostly focus in transcriptional activity. Models for post-transcriptional processes that influence the amount of gene product in a cell type specific manner are still missing. In this project, we aim at improving our ability to model complex post-transcriptional processes, such as mRNA structure, ribosome loading and RNA degradation, and finally integrate these models with models of transcriptional processes into a foundational model for regulation of gene expression. Post-transcriptional processes are affected by complex non-linear sequence interactions, and new model architectures, and feature attributions will have to be developed to learn and extract the information from them. 

### Building cell type agnostic sequence-to-function models from knock-out data

Deep sequence-to-function models learn the relationship between genomic sequence and genome-wide functional molecular measurements. Trained on data from multiple cell types, these models are capable of developing a foundational understanding of the cis-regulatory code for individual cell types. However, current architectures have to learn the cell type specific code from data of each cell type individually or in a multi-task fashion. These models only learn the direct link between sequence elements and functional molecular phenotypes but cannot reason which trans-factors are causing differences between cell types, such as gene regulatory networks. It is hoped that next-generation models will be able to use the information about the abundance of trans-acting factors to interpolate to unseen cells. These models will use readily accessible data, such as gene expression of regulatory factors, as an additional input to the model to determine the cell type and adjust the parameters of the model to interpolate to new unseen cell types. To effectively learn the relationship between factors and the cis-regulatory code, measurements in which a single or multiple regulatory factor were perturbed are essential to train the models on. This data serves as training data, as well as validation on which the models’ zero-shop ability can be tested. 


## Installation

Download the repository and setup conda environment.

Install by navigating to the location of the local repository

`pip install -e .`

## Usage

In this example, ...

You can download the model parameters from

```
mkdir data
cd data
wget https://zenodo.org/record/3402406/files/deepsea.beluga.pth
```

### Sequence attributions

See also ... 

```math
a_{local} = m_{s_0}
```
```math
a_{global} = m_{s_0} \cdot (s_0 - s_{baseline})
```

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from tism.modddels import Beluga
from tism.utils import plot_attribution, ism, deepliftshap
from tism.torch_grad import correct_multipliers, takegrad
from tangermeme.utils import random_one_hot

parameters = '../data/deepsea.beluga.pth'
model = Beluga()
model.load_state_dict(torch.load(parameters))

N=1
b=4
input_length = 2000

x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
x = substitute(x, "CTCAGTGATG")
x = x.detach().cpu().numpy()

track = 267
vis_seq = 0

grad_local = takegrad(x, model, tracks = track, output = 'local', device = None, baseline = None)
grad_local0 = grad_local[vis_seq,0][...,900:1100]
fig_local = plot_attribution(grad_local0, heatmap = grad_local0, ylabel = 'Grad\n(local)')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/Local_attributions_gradient.jpg)


<!-- <img src="https://github.com/LXsasse/TISM/blob/main/results/Comparison_time_N_cpu.jpg" width="500"> this is a comment -->

## References
<a id="1">[1]</a>
Quick and effective approximation of in silico saturation mutagenesis experiments with first-order Taylor expansion
Alexander Sasse, Maria Chikina, Sara Mostafavi,bioRxiv 2023.11.10.566588; doi: https://doi.org/10.1101/2023.11.10.566588



