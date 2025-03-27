# DRG

This repository is under develpment. The current stage may not be fully functional. Please try to fix small bugs yourself and add your contributions with pull requests.

This repository contains pytorch-based code to develop Convolutional Neural Networks (CNNs) for genomic sequence-to-function (S2F) models that predict molecular phenotypes from genomic sequence, such as gene expression or chromatin accessibility. 

It also contains scripts to visualize, compare, and assess model performance, or to compute and visualize feature importance, and perform other analysis. 

Please refer to **`examples/`** to learn how to use individual scripts and perform entire analysis. 

## Background

The cis-regulatory code engraves the details about when, where, and how much of all gene products are created. It enables multi-cellular organisms to create several cell types from a single genome by regulating transcription, processing, and degradation of gene products. Unsurprisingly, variants that disturb the information in cis-regulatory elements can result in various genetic diseases. However, measuring the impact of all cis-regulatory sequence variations to function at a cell type specific basis is combinatorially infeasible.
Deep sequence-to-function models learn the relationship between genomic sequence and genome-wide functional molecular measurements. Trained on data from multiple cell types, these models are capable of developing a foundational understanding of the cis-regulatory code for individual cell types.

## Installation

Download the repository and setup miniconda or python environment.

Install by navigating to the location of the local repository

`pip install -e .`

## Usage

Please refer to the individual examples in `examples/`

#### Process data for modeling

Please refer to the individual steps shown in [Process_data.md](https://github.com/sasselab/DRG/blob/main/examples/Process_data.md)

#### Single sequence - multi-task or multi-modal multi-task

Please refer to the individual steps shown in [Train_single_input.md](https://github.com/sasselab/DRG/blob/main/examples/Train_single_input.md)

#### Multiple sequence - multi-modal modalities

Please refer to the individual steps shown in [Train_multi_input.md](https://github.com/sasselab/DRG/blob/main/examples/Train_multi_input.md)

#### Load pre-trained models for fine-tuning, Inference

Please refer to the individual steps shown in [Finetune_models.md](https://github.com/sasselab/DRG/blob/main/examples/Finetune_models.md) and [Inference_with_models.md](https://github.com/sasselab/DRG/blob/main/examples/Inference_with_models.md)

#### Extract and interpret the learned cis-regulatory syntax

To analyze individual kernels of the model, please refer to [Kernel_analysis.md](https://github.com/sasselab/DRG/blob/main/examples/Kernel_analysis.md). To understand the sequence grammar of individual sequences, or summarize the motifs in these sequence via clustering, see [Attribution_analysis.md](https://github.com/sasselab/DRG/blob/main/examples/Attribution_analysis.md)

#### Extract and interpret the learned cis-regulatory grammar for multi-sequence input models

Please check out [Analyze_multi_input.md](https://github.com/sasselab/DRG/blob/main/examples/Analyze_multi_input.md)


## Examples

Trained example models can be downloaded for ATAC-seq data from Calderon et al 2019 from this [zenodo submission](https://zenodo.org/records/10463521).

Trained model parameters were generated with the slightly outdated example scripts here https://github.com/mostafavilabuw/Calderon2019ATACmodel/. Models were trained, assessed, and interpreted on ATAC-seq data from Calderon et al. 2019. The data count matrix with aligned ATAC-seq Tn5 cuts can be downloaded from the Gene Expression Omnibus (GEO: GSE118189) (Calderon et al. 2019). The data matrix contains 175 measurements for 829,942 genomic regions (peaks), spanning 45 unique human immune cell states from eleven different donors. The 45 immune cell states cover 25 resting cell types of which 20 are also included as stimulated cell states (Immature NK, Memory NK, Myeloid DC, Plasmablasts, and pDC don’t have a measured stimulated state).

#### Download examples
```
mkdir models
cd models

# Download model specifications
wget https://zenodo.org/records/10463521/files/ATACcountsonPeak400rcomp-cv10-1_Cormsek256l15FfGELUrcTvlCotasft101_dc5i1d1-2-4-8-16s1l7r1_tc4dNoned1s1r1l7mw3nfc3s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_model_params.dat

# Download model parameters
wget https://zenodo.org/records/10463521/files/ATACcountsonPeak400rcomp-cv10-1_Cormsek256l15FfGELUrcTvlCotasft101_dc5i1d1-2-4-8-16s1l7r1_tc4dNoned1s1r1l7mw3nfc3s512cbnoTfdo0.1tr1e-05SGD0.9bs64-F_parameter.pth
```

#### Intialize and load model for inference and interpretation

To be included

#### Get sequence attributions

Sequence attributions are derived from the linear approximations of the model in the close vicinity of individual data points. The linear approximation's feature coefficients are used to describe the impact of each feature to the model's predictions for each individual data point. Hence, each data point's attributions are described by individual linear approximations. Different sequence attribution methods use different strategies to derive the feature coefficients, using different sampling or backpropagation methods. Moreover, they are using different processing strategies to visualize and interpret the computed feature coefficients, the most common processings are called ***local*** (leave as is), ***global*** (multiply by input), and ***hypothetical*** (correct for dependencies) attributions. 

```math
a_{local} = m_{s_0}
```
```math
a_{global} = m_{s_0} \cdot (s_0 - s_{baseline})
```
```math
a_{hypo}(j)= m_{s_0}(j) - \sum_{i}^{\{A,C,G,T\}} b(i) \cdot m_{s_0}(i) \; ; \: j \in \{A,C,G,T\}
```

Sequence attributions can be determined with the model's gradient or ISM. 
Please check out [TISM](https://github.com/LXsasse/TISM/blob/main/tutorials/attribution_scores_comparison.ipynb) for more details on attribution scores, or check out one of these manuscripts [[1]](#1)[[2]](#2)


```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from tism.models import Beluga
from tism.utils import plot_attribution, ism, deepliftshap
from tism.torch_grad import correct_multipliers, takegrad
from tangermeme.utils import random_one_hot

parameters = '../data/deepsea.beluga.pth'
model = Beluga()
model.load_state_dict(torch.load(parameters))

N=1
b=4
input_length = 2000

# Generate random sequence and include motifs
x = random_one_hot((N, b, input_length), random_state = 1).type(torch.float32)
x = substitute(x, "CTCAGTGATG")
x = x.detach().cpu().numpy()

# Select output track corresponding to the motif
track = 267
vis_seq = 0

# Compute corrected gradient
grad_corrected = takegrad(x, model, tracks =track, output = 'corrected', device = None, baseline = None)

# Alternatively also referred to hypothetical attributions to the uniform baseline sequence with 0.25 probability for all bases
grad_hypothetical_uniform = takegrad(x, model, tracks = track, output = 'hypothetical', device = None, baseline = np.ones(b)*0.25)

for i in range(np.shape(x)[0]):
	print(pearsonr(grad_corrected[i].flatten(), grad_hypothetical[i].flatten())

grad_corrected0 = grad_corrected[vis_seq,0][...,900:1100]
fig = plot_attribution(grad_corrected0, heatmap = grad_corrected0, ylabel = 'corrected grad')
```
![image](https://github.com/LXsasse/TISM/blob/main/results/Corrected_gradients.jpg)

More commonly only the attribution of the base that is present in the sequence is shown to demonstrate what contributed to the prediction rather than what the model would prefer at this position.

```
grad_corrected_viz = grad_corrected[vis_seq,0][...,900:1100] *x[vis_seq][..., 900:1100]
fig = plot_attribution(grad_corrected_viz, ylabel = 'corrected grad * seq')
```

<!-- <img src="https://github.com/LXsasse/TISM/blob/main/results/Comparison_time_N_cpu.jpg" width="500"> this is a comment -->

## Recommendations and future goals

This repository is already using [captum](https://captum.ai/), and [tangermeme](https://github.com/jmschrei/tangermeme). We highly recommend checking out their documentations and [tutorials](https://github.com/jmschrei/tangermeme/tree/main/docs/tutorials) for more cool model interpretation tools. 

In the future, we are planning to incorporate more tools from [gReLU](https://github.com/Genentech/gReLU) to speed up model training and analysis. For a fast start, please check out their tutorials. 

## References
<a id="1">[1]</a> 
Quick and effective approximation of in silico saturation mutagenesis experiments with first-order Taylor expansion
Alexander Sasse, Maria Chikina, Sara Mostafavi,bioRxiv 2023.11.10.566588; doi: https://doi.org/10.1101/2023.11.10.566588 

<a id="2">[2]</a>
Majdandzic, A., Rajesh, C. & Koo, P.K. Correcting gradient-based interpretations of deep neural networks for genomics. Genome Biol 24, 109 (2023). https://doi.org/10.1186/s13059-023-02956-3

