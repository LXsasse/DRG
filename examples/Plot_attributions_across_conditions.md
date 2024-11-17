# Plot attributions for a set of well predicted sequences with strong signal

Create a file with data point names that one wants to look at. 

```
tset=../exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclPBSex_Max4.6_MaxDiff1.0_00.5sim_list.txt
tset=../pbslist.txt
#tset=exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pnt_corr_tclBfoexlogp10_fc1.0_p1.301_00.5sim_list.txt
```

Get the attributions to the set of sequences
```
atts=exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_gradPBSs2.npz
#atts=exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_gradBfo.npz
```
i
```
list=$(cat $tset)
cell=PBS2

for gene in $list
do
echo $l

python ~/Scripts/DRG/plot_acrosscells_attribution_maps.py $atts ../TSS40k.npz $gene all 0 -remove_low_attributions 0.75 --dpi 60 --outname Attributionplots/${gene}_${cell}_ex_attributions_TSS40 --add_predictions exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pred.npz --centerofmass_attributions 100 --vlim -0.7,0.7 --add_measured ../exonic.tsv --add_measured_apendix _ex

python ~/Scripts/DRG/plot_acrosscells_attribution_maps.py $atts ../RNA40k.npz $gene all 1 -remove_low_attributions 0.75 --dpi 60 --outname Attributionplots/${gene}_${cell}_ex_attributions_RNA40 --add_predictions exonicdegradintroniconTSS40kRNA40krcomp_sd1-cv10-1_Cormsek300l15TfGELUwei4rcTvlCotaNone_dc3i1d1-2-4s1l11r1_tc4d300d1s1r1l11mw6nfc3s1024dicedictdictcbnoTfdo0.1tr1e-05SGD0.9bs8-F_comb88nl0Linear1.1r0_pred.npz --centerofmass_attributions 100 --vlim -0.7,0.7 --add_measured ../exonic.tsv --add_measured_apendix _ex
done

```
