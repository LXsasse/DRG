# Systematically analyze attributions from a model

## Plot attributions of single sequence

```
# For example
$indiv_track_grad=M_gradall.npz # gradients for each output track
$celltype_avg_grad=M_gradavgcell.npz # gradients averaged for each cell type
$modality_avg_grad=M_gradmod.npz # gradients averaged for each modality

# One-hot encoded sequences that were used to generate the attributions
$input_seqs=seq2k.npz

# Select the sequence you want to look at
$selected_sequence=0
$selected_sequence=ENSG0000134202_GSTM3 # can also be the name of sequence

# For models with two inputs, add another index for the sequence that is used from the attributions after argument for $tracks

$Tcelltracks='musthave=T8' # if you want to select all T8-cells
$PBStracks='musthave=PBS' # if you want to select all cells in control PBS
$tracks=0,1,5,10 # if you know the indices of your tracks
$tracks=T8_IL2_ex,T8_IL7_ex,T8_IL_ex # if you know the names of your tracks

python /path/to/DRG/scripts/sequence_attributions/run_plot_acrosstracks_attribution_maps.py $indiv_track_grad $input_seqs $selected_sequence $Tcelltracks 0 --remove_low_attributions 0.5 --dpi 50
# --remove_low_attributions removes attributions with less than 0.5 of the max across the sequence from the flanks of the attributions.
```


## Extract motif seqlets from all sequence attributions in the attributions file

For the multi-sequence input model, one might have to split the attributions from both sequences manually first. 

```
python DRG/scripts/sequence_attributions/run_extract_motifs_from_attributionmaps.py $modality_avg_grad $input_seqs 1.96 1 4 global --select_tracks all

seqleteffects=${modality_avg_grad%.npz}globalmotifs1.96_1_4.txt
seqlets=${modality_avg_grad%.npz}globalmotifs1.96_1_4.npz
```

## Cluster the hundreds of thousands extracted seqlets
Computes a similarity matrix between extracted seqlets and clusters them with agglomerative clustering and distance threshold.
```
python DRG/scripts/interpret_models/compute_pwm_correlation_cluster_and_combine.py ${seqlets} complete --distance_threshold 0.05 --distance_metric correlation_pvalue 

seqlet_clusters=${seqlets%.npz}_cldcomplete0.05corpva.txt
seqlet_clustermotifs=${seqlets%.npz}_cldcomplete0.05corpvapfms.meme
```

## Generate motif effect matrix: sequence versus cluster effect

Determine the activity of motifs clusters in each sequence
```
python DRG/scripts/sequence_attributions/create_sequence_motif_matrices.py $seqlet_clsuters $seqleteffects --minimum_size 1 --motif_statistic max --outname ${seqlet_clusters%txt} 
```

Can also average over sequence clusters, use --sequence_clusters and provide file
```
--sequence_clusters motifeffects=${seqlet_clusters%txt}_seqmatmaxms1.txt 
```

## Plot Effect matrix
```
python DRG/scripts/interpret_models/plot_pwm_in_tree.py $seqlet_clustermotifs --heatmap va_seqmatmaxdiffATACms3.txt --reverse_complement 
```








