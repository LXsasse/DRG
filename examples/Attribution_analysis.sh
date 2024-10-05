# Plot attributions of single sequence

$indiv_track_grad=M_gradall.npz # gradients for each output track
$celltype_avg_grad=M_gradavgcell.npz # gradients averaged for each cell type
$modality_avg_grad=M_gradmod.npz # gradients averaged for each modality

$input_seqs=seq2k.npz

$selected_sequence=0
$selected_sequence=ENSG0000134202_GSTM3 # can also be the name of sequence

# For models with two inputs add another index for the sequence that is used after tracks

$Tcelltracks='musthave=T8' # if you want to select all T8-cells
$PBStracks='musthave=PBS' # if you want to select all cells in control PBS
$tracks=0,1,5,10 # if you know the indices of your tracks
$tracks=T8_IL2_ex,T8_IL7_ex,T8_IL_ex # if you know the names of your tracks
python /DRG/scripts/sequence_attributions/run_plot_acrosstracks_attribution_maps.py $indiv_track_grad $input_seqs $selected_sequence $Tcelltracks 0 --remove_low_attributions 0.5 --dpi 50


# Extract motfis from all sequence attributions in the attributions file
# Might have to split the attributions from both sequences first
python DRG/scripts/sequence_attributions/run_extract_motifs_from_attributionmaps.py $modality_avg_grad $input_seqs 1.96 1 4 global --select_tracks all

seqleteffects=${modality_avg_grad%.npz}globalmotifs1.96_1_4.txt
seqlets=${modality_avg_grad%.npz}globalmotifs1.96_1_4.npz
# cluster seqlets the hundreds of thousands extracted seqlets
# We might have to change the clustering (i.e. something that runs well with huge numbers of data points), and the similarity computation (i.e. only record similarity if better than cutoff, potentially return sparse graph for graph clustering).

python DRG/scripts/interpret_models/compute_pwm_correlation_cluster_and_combine.py ${seqlets} complete --distance_threshold 0.05 --distance_metric correlation_pvalue 
seqlet_clusters=${seqlets%.npz}_cldcomplete0.05corpva.txt
seqlet_clustermotifs=${seqlets%.npz}_cldcomplete0.05corpvapfms.meme

# Generate sequence to cluster effect matrix
python DRG/scripts/sequence_attributions/create_sequence_motif_matrices.py $seqlet_clsuters $seqleteffects --minimum_size 1 --motif_statistic max --outname ${seqlet_clusters%txt} 
# can also average over sequence clusters, use --sequence_clusters and provide file
motifeffects=${seqlet_clusters%txt}_seqmatmaxms1.txt 
# Plot Effect matrix
python DRG/scripts/interpret_models/plot_pwm_in_tree.py $seqlet_clustermotifs --heatmap va_seqmatmaxdiffATACms3.txt --reverse_complement 









