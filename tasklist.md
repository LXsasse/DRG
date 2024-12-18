# Tasklists

## Repository organization

- [x] Split code between drg_tools and scripts that use these tools
- [x] Write short tutorials on how to use modules in drg_tools in examples
- [ ] Write example .mds on how to use scripts
- [ ] Write bpmodel training on bias code

## Model training

- [ ] Implement sequence shifting from left to right for transcripts
- [ ] Remove params0.pth and find different way to return to same parameters with random intialization
- [ ] Automatically set optimizer to zero after gradient explosion
- [ ] Include target mask for data points for every modality
- [ ] Include superconvergence learning rate https://arxiv.org/abs/1708.07120
- [ ] Use predicted information in a latter layer and pass it back to an earlier layer 3 times.
- This is similar to the alpha fold techniques which seem like a new type of recurrent block, a specialized recurrent block.
- [ ] Predict confidence and include unlabeled data points with predicted outputs. 
- [ ] Try to use automatic gradient descent https://github.com/jxbz/agd


## Model architectures

- [ ] Create common sequence_embedding_model for cnn_model.py and bpcnn_model.py that is shared between the two networks and identical up to the last layer.
Introduce learnable cell type and data modality vector.
- [ ] Position invariant convblocks and training for those
- To capture these relative distances between motifs we need long-range equidistant modules
- [ ] Allow different conv. blocks for different input sequences
- [ ] Work on extended Non-linear convolution model, so that additional layers are not used for motif improvement (i.e. base-pair interactions) but for motif interactions
- [ ] Improve long-range convolutions, such as hyena
- [ ] Check kwargs handling in model_params.dat file


## Model interpretation 

- [x] Enable DeepLIFT for multiple sequence inputs
- [x] DeepLIFT to multiple random seeds
- [ ] Find motifs in attributions with sequence specific cut off from dinuc shuffle
- [ ] Write module that can extract sequences from 'sparse'-attribution maps with positions
- [ ] Add motif annotations to plot attributions, with motif data base and conv scanning, and with names and location file

