# Analyze the learned kernels of model for regulatory motifs
# Kernel weights are extracted
kerweight=*_kernelweights.meme
input_sequences=seq2k.npz # One-hot encoded sequences
# If you have two sets of sequences as model input, split _kernelweights.meme beforehand and into the kernels from the two CNNs, and transform the kernels with their respective sequences that they see during training

# Transform kernel weights to motifs
python ../DRG/scripts/interpret_models/transform_kernels_to_pwms.py $kerweight --activated EXP --maxactivation 0.5 --batchsize 10 --sequences $input_sequences --reverse_complement --nrandom 1000 --device 'cpu'
# --nrandom selects 1000 random sequences from seq2k.npz. Each sequence contains rougly L seqlets, so that the motifs are generated from LXnrandom seqlets. 
# > Select nrandom so that ~100Mio seqlets are used to generate motifs
# This returns a motif file with the name suffixes: _seqset20.0KrcEXPmax0.5_kernel_pwms.meme

# If you have multiple kernel motifs from different initializations of your model (normally 5-10 replicates) then combine the motifs, cluster them and see how reproducible the motifs were found.
pwms='*m0_seqset20.0KrcEXPmax0.5_kernel_pwms.meme m1_seqset20.0KrcEXPmax0.5_kernel_pwms.meme m3_seqset20.0KrcEXPmax0.5_kernel_pwms.meme' # three kernel motif files as examples
allpwms=m0to3_seqset20.0KrcEXPmax0.5_kernel_pwms.meme 
python ../DRG/scripts/interpret_models/write_pwms_from_multfiles_tofile.py $pwms mh0,mh1,mh2,mh3 $allpwms
# Cluster kernel motifs
python ../DRG/scripts/interpret_models/compute_pwm_correlation_cluster_and_combine.py $outname complete --distance_threshold 0.05 --infocont --reverse_complement --distance_metric correlation_pvalue --clusternames
# Cluster assignments are stored in this file
clusters=${allpwms%.meme}ms4ic_cldcomplete0.05corpva.txt

kerneffectmats='m0_kerneffct.dat m1_kerneffct.dat m2_kerneffct.dat'
# Use cluster assignments based on motif similarities to combine kernel effect matrices from different model initializations
# If kernel effects do not sufficiently correlate with each other, the motif cluster will be split up, i.e. a new final cluster file will be generated from motif clusters and similarities of effects
python DRG/scripts/interpret_models/subcluster_and_combine_pwm_importance_matrices.py $clusters $kerneffectmats 

final_clusters=${clusters%.txt}_clkerneffct.txt
final_effects=${clusters%.txt}_clmkerneffct.dat

# Check reproducibility
python DRG/scripts/interpret_models/pwm_cluster_stats.py $final_clusters
#returns
prod=${final_clusters%.meme}_clkerneffct_reprod.txt
# Plot histogram for reproducibility
python DRG/scripts/data_analysis/plot_data_histogram.py $prod --bins 0.5,10.5,11 --xlabel "Number models in cluster" --addcumulative -1 --outname ${prod%.txt} 

# Use the these final cluster assignments and generate the combined motif pwms
python DRG/scripts/interpret_models/compute_pwm_correlation_cluster_and_combine.py $allpwms $final_clusters --reverse_complement --infocont --distance_metric correlation_pvalue --clusternames

# Normalize combined cluster motifs to use with Tomtom
python ~/Scripts/Git/DRG/scripts/interpret_models/parse_motifs_tomeme.py ${final_clusters}pfms.meme --strip 0.25 --norm

# Compare motifs to motif data base
motif_database=mouse_pfms_v4.meme
tomtom -thresh 0.1 -dist pearson -text ${final_clusters}pfmsnrm.meme $motif_database > ${final_clusters%.meme}pfms.tomtom.tsv
tomtom=${final_clusters%.meme}pfms.tomtom.tsv

# Assign TFs names to kernel motifs based on motif match
python DRG/scripts/interpret_models/replace_pwmname_with_tomtom_match.py $tomtom q 0.05 ${final_clusters}pfms.meme --only_best 10 --split_tomtomnames '_' 2 --generate_namefile -reduce_clustername Number -usepwmid --outname ${final_clusters%.meme}

# Plot kernel cluster effects with kernel motifs, sorted in tree
repcut=3 # Minimum reproducible cutoff to show motif
python DRG/scripts/interpret_models/plot_pwm_in_tree.py ${final_clusters}pfms.meme --setcut $prod $repcut --heatmap $final_effects --pwmnames ${final_clusters%.meme}q0.05best10_altnames.txt --reverse_complement --savefig ${final_effects}.rp${repcut} --sortx None --dpi 250 --infocontpwms --start_heatmap 2

# Chose one row, i.e. one motif and plot the effects in a line plot
k=12
#python $drgdir/interpret_models/plot_kernel_activity_across.py $final_effects $k --split_matrix '.' --join_columns ../../CutandRun_and_ATAC.lineages.txt --split_matrix_norm --outname ${effects%.dat} --ylim -1 1

# Plot individual logos of kernels
#python $drgdir/interpret_models/plot_pwm_logos.py $pfms --select $k --replace_name_with_id --infocont

# Instead of effects, we can also perform this anlysis for other global values of the kernels, for example the delta correlation matrix, which measures the influence to the correlation across tracks in a testclass
## We just need to perform the subclustering on the matrices from each initialization first, or use the subclusters from the effect_matrix and combine these matrices the same way, setting --mincorr to 2.
deltacorr=${clusters%txt}_clmkernattmeantestclass0.3.dat
python DRG/scripts/interpret_models/plot_pwm_in_tree.py ${final_clusters}pfms.meme --setcut $prod $repcut --heatmap $deltacorr --pwmnames ${final_clusters%.meme}q0.05best10_altnames.txt --reverse_complement --savefig ${final_effects}.rp${repcut} --sortx None --dpi 250 --infocontpwms --start_heatmap

# Additionally to assessing the effects for individual output tracks, we can also look at the effects of the motif for each data modality but summarizing the effects from the effect matrices for each data modality
# 1) Split kernel effects based on header into matrices for each data modality, and then compute the mean, coefficient of variation, scaledmean, max, and scaled max, scaled to the largest absolute effect in the data modality 






