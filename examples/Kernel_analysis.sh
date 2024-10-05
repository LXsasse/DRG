# Transform kernel weights to motifs
kerweight=KER-cv10-1fromFTBmh3-cv10-1_Cormsek512l19TfEXPGELUmax10rcTvlCota_tc2dNoned1s1r1l7ma5nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs64-F_kernelweights.meme
input_sequences=seq2k.npz # One-hot encoded sequences
# If you have two sets of sequences, maybe split kernelweights.meme beforehand and run the kernels with the respective sequences that they see during training
python ~/Scripts/Git/DRG/scripts/interpret_models/transform_kernels_to_pwms.py $kerweight --activated EXP --maxactivation 0.5 --batchsize 10 --sequences seq2k.npz --reverse_complement -unique --nrandom 1000 --device 'cpu'
# --nrandom selects 1000 random sequences from seq2k.npz. Each sequence contains rougly L seqlets, so that the motifs are generated from LXnrandom seqlets. 
# > Select nrandom so that ~100Mio seqlets are used to generate motifs

# Cluster kernel motifs
# Once you have the kernel motifs from different intializations, cluster them
KER-cv10-1fromFTBmh0-cv10-1_Cormsek512l19TfEXPGELUmax10rcTvlCota_tc2dNoned1s1r1l7ma5nfc3s1024cbnoTfdo0.1tr1e-05SGD0.9bs64-F_seqset20.0KrcEXPmax0.5_kernel_pwms.meme
python ~/Scripts/Git/DRG/scripts/interpret_models/compute_pwm_correlation_cluster_and_combine.py $outname complete --distance_threshold 0.01 --infocont --reverse_complement --distance_metric correlation_pvalue --clusternames --outname ${oname%.meme}ms4ic





