# BiomeNED
This package is the code to accompany our paper  "Deep in the Bowel: Highly Interpretable Neural Encoder-Decoder Networks Predict Gut Metabolites from Gut Microbiome" - https://www.biorxiv.org/content/10.1101/686394v1.abstract

The results reported in the paper can be reproduced by running script main_cv_1dir.py
For example, for the Nonneg-Sparse-NED performance reported in table 3, run:
python main_cv_1dir.py --model BiomeAESnip --sparse 0.06 --learning_rate 0.01 --batch_size 20 --latent_size 70 --activation "tanh_tanh" --data_type "clr" --nonneg_weight --normalize_input --draw_graph
This script will also generate the computational graph shown in the paper.

Setup requirement:
This code runs on python 3.6 or newer on Anaconda 3 environment.
Main dependencies:
pytorch 1.1.0
scikit-learn 0.20.1 
graphviz 0.10.1
matplotlib 3.0.2
